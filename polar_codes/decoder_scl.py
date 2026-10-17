"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
    _channel_llr_to_decoder,
    _update_bits,
    _update_llrs,
    sc_decode,
)

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for b in bits:
        reg = ((reg << 1) | int(b)) & mask
        if reg & top:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class _Path:
    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n, llr_dec):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_dec


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _pm_penalty(self, llr, u_bit):
        u_from_llr = 0 if llr >= 0 else 1
        return abs(llr) if u_bit != u_from_llr else 0.0

    def decode(self, llr_ch):
        if self.list_size == 1:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_dec = _channel_llr_to_decoder(np.asarray(llr_ch, dtype=np.float64))
        paths = [_Path(self.N, self.n, llr_dec)]

        for i in range(self.N):
            l = _bit_reversed_index(i, self.n)
            candidates = []

            for path in paths:
                _update_llrs(path.L, path.B, l, self.n)
                llr_root = path.L[l, self.n]

                if l in self.frozen_set:
                    new_path = _Path(self.N, self.n, llr_dec)
                    new_path.pm = path.pm + self._pm_penalty(llr_root, 0)
                    new_path.L = path.L.copy()
                    new_path.B = path.B.copy()
                    new_path.B[l, self.n] = 0
                    _update_bits(new_path.B, l, self.n, self.N)
                    candidates.append(new_path)
                else:
                    for u_bit in (0, 1):
                        new_path = _Path(self.N, self.n, llr_dec)
                        new_path.pm = path.pm + self._pm_penalty(llr_root, u_bit)
                        new_path.L = path.L.copy()
                        new_path.B = path.B.copy()
                        new_path.B[l, self.n] = u_bit
                        _update_bits(new_path.B, l, self.n, self.N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best = paths[0]
        u_hat = best.B[:, self.n].astype(int)

        if self.crc_length > 0:
            passing = []
            for p in paths:
                uh = p.B[:, self.n].astype(int)
                if crc_check(uh[self.info_indices], self.crc_length):
                    passing.append(p)
            if passing:
                best = min(passing, key=lambda p: p.pm)
                u_hat = best.B[:, self.n].astype(int)

        return u_hat, best.pm
