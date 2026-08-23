"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _upper_llr,
    _lower_llr,
    _hard_decision,
)


def _run_crc_bits(data_bits, crc_length):
    if crc_length == 8:
        poly = 0x07
        mask = 0xFF
    else:
        poly = 0x8005
        mask = 0xFFFF
    reg = 0
    for bit in data_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    crc_bits = _run_crc_bits(info_bits, crc_length)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    payload = bits[:-crc_length]
    expected = _run_crc_bits(payload, crc_length)
    return np.array_equal(bits[-crc_length:], expected)


class _Path:
    def __init__(self, N, n, llrs):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llrs
        self.u = np.zeros(N, dtype=int)

    def copy(self):
        p = object.__new__(_Path)
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.u = self.u.copy()
        p.pm = self.pm
        return p


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = (
            np.asarray(info_indices, dtype=int) if info_indices is not None else None
        )

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            bs = 2 ** (s + 1)
            br = bs // 2
            for j in range(l, self.N, bs):
                if j % bs < br:
                    path.L[j, s + 1] = _upper_llr(path.L[j, s], path.L[j + br, s])
                else:
                    path.L[j, s + 1] = _lower_llr(
                        path.L[j - br, s],
                        path.L[j, s],
                        path.B[j - br, s + 1],
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            bs = 2 ** s
            br = bs // 2
            for j in range(l, -1, -bs):
                if j % bs >= br:
                    path.B[j - br, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - br, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, u):
        v = 0 if llr >= 0 else 1
        return 0.0 if u == v else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1:
            from decoder_sc import sc_decode

            return sc_decode(llr_ch, self.frozen_bits), 0.0

        paths = [_Path(self.N, self.n, llr_ch)]

        for l in [_bit_reversed(i, self.n) for i in range(self.N)]:
            new_paths = []
            for path in paths:
                self._update_llrs(path, l)
                llr_bit = path.L[l, self.n]
                if l in self.frozen_set:
                    path.u[l] = 0
                    path.B[l, self.n] = 0
                    path.pm += self._pm_penalty(llr_bit, 0)
                    self._update_bits(path, l)
                    new_paths.append(path)
                else:
                    for u in (0, 1):
                        cp = path.copy()
                        cp.u[l] = u
                        cp.B[l, self.n] = u
                        cp.pm += self._pm_penalty(llr_bit, u)
                        self._update_bits(cp, l)
                        new_paths.append(cp)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[:self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                bits = (
                    p.u[self.info_indices]
                    if self.info_indices is not None
                    else p.u
                )
                if crc_check(bits, self.crc_length):
                    valid.append(p)
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)
        return best.u.copy(), best.pm

