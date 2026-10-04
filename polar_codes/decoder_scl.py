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
    _logdomain_sum,
    sc_decode,
    _frozen_mask,
)

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


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
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class _SCPath:
    __slots__ = ("L", "B", "pm")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.pm = 0.0

    def copy(self):
        p = _SCPath.__new__(_SCPath)
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        return p


class SCLDecoder:
    """SCL 译码器（基于 Permuted SCD 的多路径扩展）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = _frozen_mask(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen)[0]

    def _upper_llr(self, l1, l2):
        if np.isinf(l1) and not np.isinf(l2):
            return l2
        if not np.isinf(l1) and np.isinf(l2):
            return l1
        if np.isinf(l1) and np.isinf(l2):
            return np.inf
        return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)

    def _lower_llr(self, l1, l2, b):
        if b == 0:
            if np.isinf(l1) or np.isinf(l2):
                return np.inf
            return l1 + l2
        return l1 - l2

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = self._upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = self._lower_llr(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        int(path.B[j - branch_size, s + 1]),
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen), 0.0

        paths = [_SCPath(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            new_paths = []
            for path in paths:
                self._update_llrs(path, l)
                llr0 = path.L[l, self.n]
                if self.frozen[l]:
                    cand = path.copy()
                    penalty = abs(llr0) if llr0 < 0 else 0.0
                    cand.pm += penalty
                    cand.B[l, self.n] = 0
                    self._update_bits(cand, l)
                    new_paths.append(cand)
                else:
                    for bit in (0, 1):
                        cand = path.copy()
                        llr_bit = 0 if llr0 >= 0 else 1
                        if bit != llr_bit:
                            cand.pm += abs(llr0)
                        cand.B[l, self.n] = bit
                        self._update_bits(cand, l)
                        new_paths.append(cand)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                bits = p.B[:, self.n].astype(int)
                if crc_check(bits[self.info_indices], self.crc_length):
                    valid.append(p)
            chosen = min(valid, key=lambda p: p.pm) if valid else min(paths, key=lambda p: p.pm)
        else:
            chosen = min(paths, key=lambda p: p.pm)

        return chosen.B[:, self.n].astype(int), chosen.pm
