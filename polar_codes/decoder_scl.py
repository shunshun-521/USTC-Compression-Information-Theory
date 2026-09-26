"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _lower_llr,
    _update_bits,
    _update_llr,
    _upper_llr,
    align_channel_llr,
    sc_decode,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    poly = _crc_poly(crc_length)
    reg = 0
    bits = np.asarray(info_bits, dtype=int).ravel()
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array([(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([bits, crc_bits])


def crc_check(bits, crc_length=8):
    if crc_length == 0:
        return True
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in np.asarray(bits, dtype=int).ravel():
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class _Path:
    __slots__ = ("L", "B", "pm")

    def __init__(self, n, N, llr_internal):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_internal
        self.pm = 0.0


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.frozen_set = set(np.where(self.frozen_bits)[0])

    def _pm_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        llr_internal = align_channel_llr(llr_ch)
        paths = [_Path(self.n, self.N, llr_internal)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            new_paths = []
            for path in paths:
                _update_llr(path.L, path.B, l, self.n, self.N)
                llr = path.L[l, self.n]
                if l in self.frozen_set:
                    path.pm += self._pm_penalty(llr, 0)
                    path.B[l, self.n] = 0
                    _update_bits(path.B, l, self.n, self.N)
                    new_paths.append(path)
                else:
                    for u in (0, 1):
                        p2 = _Path(self.n, self.N, llr_internal)
                        p2.L = path.L.copy()
                        p2.B = path.B.copy()
                        p2.pm = path.pm + self._pm_penalty(llr, u)
                        p2.B[l, self.n] = u
                        _update_bits(p2.B, l, self.n, self.N)
                        new_paths.append(p2)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        candidates = [(p.pm, p.B[:, self.n].astype(int)) for p in paths]
        if self.crc_length > 0:
            ok = [(pm, u) for pm, u in candidates if crc_check(u[self.info_indices], self.crc_length)]
            pm, u_hat = min(ok if ok else candidates, key=lambda x: x[0])
        else:
            pm, u_hat = min(candidates, key=lambda x: x[0])
        return u_hat, pm
