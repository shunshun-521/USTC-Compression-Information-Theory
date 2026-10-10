"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _upper_llr_exact,
    _lower_llr_exact,
    _frozen_index_set,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def _crc_remainder(bits, crc_length):
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    padded = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    rem = _crc_remainder(padded, crc_length)
    crc_bits = np.array([(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=np.int8)
    return _crc_remainder(bits, crc_length) == 0


class _Path:
    __slots__ = ("L", "B", "pm")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.pm = 0.0

    def copy(self):
        p = _Path.__new__(_Path)
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        return p


class SCLDecoder:
    """SCL 译码器（路径复制；L 较小时足够高效）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = _frozen_index_set(frozen_bits)
        self.L_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block = 1 << (s + 1)
            branch = block >> 1
            for j in range(l, self.N, block):
                if j % block < branch:
                    path.L[j, s + 1] = _upper_llr_exact(path.L[j, s], path.L[j + branch, s])
                else:
                    path.L[j, s + 1] = _lower_llr_exact(
                        path.L[j, s], path.L[j - branch, s], path.B[j - branch, s + 1]
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block = 1 << s
            branch = block >> 1
            for j in range(l, -1, -block):
                if j % block >= branch:
                    path.B[j - branch, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch, s])
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, u_bit):
        v = 0 if llr >= 0 else 1
        return 0.0 if u_bit == v else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            candidates = []
            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]
                if l in self.frozen_set:
                    pen = self._pm_penalty(llr, 0)
                    p2 = path.copy()
                    p2.pm += pen
                    p2.B[l, self.n] = 0
                    self._update_bits(p2, l)
                    candidates.append(p2)
                else:
                    for u_bit in (0, 1):
                        p2 = path.copy()
                        p2.pm += self._pm_penalty(llr, u_bit)
                        p2.B[l, self.n] = u_bit
                        self._update_bits(p2, l)
                        candidates.append(p2)
            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.L_size]

        best_crc = None
        best_any = min(paths, key=lambda p: p.pm)
        if self.crc_length > 0:
            for p in paths:
                u_hat = p.B[:, self.n].astype(int)
                payload = u_hat[self.info_indices]
                if crc_check(payload, self.crc_length):
                    if best_crc is None or p.pm < best_crc.pm:
                        best_crc = p
        chosen = best_crc if best_crc is not None else best_any
        u_hat = chosen.B[:, self.n].astype(int)
        return u_hat, chosen.pm
