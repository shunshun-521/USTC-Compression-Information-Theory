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
    _update_llrs,
    _upper_llr,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class _Path:
    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch.copy()

    def copy(self):
        p = _Path.__new__(_Path)
        p.pm = self.pm
        p.L = self.L.copy()
        p.B = self.B.copy()
        return p


class SCLDecoder:
    """Permuted SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.phases = [_bit_reversed(i, self.n) for i in range(N)]

    @staticmethod
    def _pm_penalty(llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for l in self.phases:
            expanded = []
            for path in paths:
                _update_llrs(path.L, path.B, l, self.n)
                llr = path.L[l, self.n]
                if self.frozen_bits[l]:
                    child = path.copy()
                    child.pm += self._pm_penalty(llr, 0)
                    child.B[l, self.n] = 0
                    _update_bits(child.B, l, self.n)
                    expanded.append(child)
                else:
                    for u in (0, 1):
                        child = path.copy()
                        child.pm += self._pm_penalty(llr, u)
                        child.B[l, self.n] = u
                        _update_bits(child.B, l, self.n)
                        expanded.append(child)

            expanded.sort(key=lambda p: p.pm)
            paths = expanded[: self.list_size]

        paths.sort(key=lambda p: p.pm)
        if self.crc_length > 0:
            info_mask = ~self.frozen_bits
            for p in paths:
                bits = p.B[:, self.n].astype(np.int8)
                ib = bits[info_mask]
                if crc_check(ib, self.crc_length):
                    return bits, p.pm
        best = paths[0]
        u_hat = best.B[:, self.n].astype(np.int8)
        return u_hat, best.pm
