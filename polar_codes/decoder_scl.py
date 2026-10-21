"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    _SCDState,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _upper_llr,
    _lower_llr,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """CRC 附加到信息比特末尾"""
    poly = _crc_poly(crc_length)
    reg = 0
    bits = np.asarray(info_bits, dtype=int).tolist()
    for b in bits:
        reg ^= (b << (crc_length - 1))
        for _ in range(8 if crc_length == 8 else 16):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]
    return np.array(bits + crc_bits, dtype=int)


def crc_check(bits, crc_length=8):
    """校验 CRC"""
    poly = _crc_poly(crc_length)
    reg = 0
    for b in np.asarray(bits, dtype=int):
        reg ^= (b << (crc_length - 1))
        for _ in range(8 if crc_length == 8 else 16):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        p = _Path.__new__(_Path)
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        p.u_hat = self.u_hat.copy()
        return p

    def update_llrs(self, l, n):
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.L.shape[0], block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = _upper_llr(self.L[j, s], self.L[j + branch_size, s])
                else:
                    top_bit = self.B[j - branch_size, s + 1]
                    self.L[j, s + 1] = _lower_llr(
                        self.L[j, s], self.L[j - branch_size, s], int(top_bit)
                    )

    def update_bits(self, l, n, N):
        if l < N / 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(
                        self.B[j - branch_size, s]
                    )
                    self.B[j, s - 1] = self.B[j, s]

    def current_llr(self, l, n):
        self.update_llrs(l, n)
        return self.L[l, n]


class SCLDecoder:
    """SCL 译码器（路径复制实现）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _pm_add(self, pm, llr, u_bit):
        v = 0 if llr >= 0 else 1
        if u_bit != v:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            new_paths = []
            for path in paths:
                llr = path.current_llr(l, self.n)
                if l in self.frozen_set:
                    p = path.copy()
                    p.pm = self._pm_add(p.pm, llr, 0)
                    p.B[l, self.n] = 0
                    p.u_hat[l] = 0
                    p.update_bits(l, self.n, self.N)
                    new_paths.append(p)
                else:
                    for u_bit in (0, 1):
                        p = path.copy()
                        p.pm = self._pm_add(p.pm, llr, u_bit)
                        p.B[l, self.n] = u_bit
                        p.u_hat[l] = u_bit
                        p.update_bits(l, self.n, self.N)
                        new_paths.append(p)
            new_paths.sort(key=lambda x: x.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if self._crc_ok(p.u_hat)]
            if valid:
                paths = valid
        best = min(paths, key=lambda x: x.pm)
        return best.u_hat.copy(), best.pm

    def _crc_ok(self, u_hat):
        info_idx = np.where(~self.frozen_bits)[0]
        payload = u_hat[info_idx]
        return crc_check(payload, self.crc_length)
