"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _lower_llr,
    _update_bits,
    _update_llrs,
    bit_reversed,
    prepare_channel_llr,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class _Path:
    __slots__ = ("L", "B", "pm")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.pm = 0.0


class SCLDecoder:
    """SCL 译码器（基于 Permuted SCD 树）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _pm_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = prepare_channel_llr(llr_ch)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            new_paths = []

            for path in paths:
                _update_llrs(path.L, path.B, l, self.n, self.N)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    child = _Path(self.N, self.n, llr_ch)
                    child.L[:] = path.L
                    child.B[:] = path.B
                    child.pm = path.pm + self._pm_penalty(llr, 0)
                    child.B[l, self.n] = 0
                    _update_bits(child.B, l, self.n, self.N)
                    new_paths.append(child)
                else:
                    for u in (0, 1):
                        child = _Path(self.N, self.n, llr_ch)
                        child.L[:] = path.L
                        child.B[:] = path.B
                        child.pm = path.pm + self._pm_penalty(llr, u)
                        child.B[l, self.n] = u
                        _update_bits(child.B, l, self.n, self.N)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        paths.sort(key=lambda p: p.pm)
        if self.crc_length > 0:
            for path in paths:
                info_bits = path.B[:, self.n].astype(int)[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    return path.B[:, self.n].astype(int), path.pm

        best = paths[0]
        return best.B[:, self.n].astype(int), best.pm
