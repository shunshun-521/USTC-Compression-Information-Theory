"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _permute_channel_llr,
    _update_bits,
    _update_llrs,
    bit_reversed,
)
from encoder import bit_reversal_permutation

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
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
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.decode_order = [bit_reversed(i, self.n) for i in range(N)]

    def _metric_penalty(self, llr_val, u_bit):
        u_from_llr = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == u_from_llr else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = _permute_channel_llr(np.asarray(llr_ch, dtype=np.float64))
        paths = [_Path(self.N, self.n, llr_ch)]

        for phi_nat, l in enumerate(self.decode_order):
            new_paths = []
            for path in paths:
                _update_llrs(path.L, path.B, l, self.n)
                cur_llr = path.L[l, self.n]
                if self.frozen_bits[l]:
                    pen = self._metric_penalty(cur_llr, 0)
                    child = _Path(self.N, self.n, llr_ch)
                    child.L[:] = path.L
                    child.B[:] = path.B
                    child.pm = path.pm + pen
                    child.B[l, self.n] = 0
                    _update_bits(child.B, l, self.n)
                    new_paths.append(child)
                else:
                    for u_bit in (0, 1):
                        pen = self._metric_penalty(cur_llr, u_bit)
                        child = _Path(self.N, self.n, llr_ch)
                        child.L[:] = path.L
                        child.B[:] = path.B
                        child.pm = path.pm + pen
                        child.B[l, self.n] = u_bit
                        _update_bits(child.B, l, self.n)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.B[:, self.n].astype(int)[self.info_indices], self.crc_length)
            ]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)
        u_hat = best.B[:, self.n].astype(int)
        return u_hat, best.pm
