"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation, bit_reversed
from decoder_sc import (
    active_bit_level,
    active_llr_level,
    lower_llr,
    sc_decode,
    upper_llr,
)

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_bits(info_bits, poly, crc_length):
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)]


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    crc = np.array(_crc_bits(info_bits, poly, crc_length), dtype=np.int8)
    return np.concatenate([info_bits, crc])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = 0
    for b in bits:
        rem ^= int(b) << (crc_length - 1)
        for _ in range(1):
            if rem & (1 << (crc_length - 1)):
                rem = ((rem << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                rem = (rem << 1) & ((1 << crc_length) - 1)
    return rem == 0


class _Path:
    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n, llr_internal):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.L[:, 0] = llr_internal


class SCLDecoder:
    """SCL 译码器（路径分裂时复制 L/B）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        frozen_bits = np.asarray(frozen_bits)
        self.frozen_set = set(np.where(frozen_bits != 0)[0].tolist())
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)
        self.inv_br = np.argsort(self.br)

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            frozen_arr = np.array([1 if i in self.frozen_set else 0 for i in range(self.N)], dtype=int)
            return sc_decode(llr_ch, frozen_arr), 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_internal = llr_ch[self.inv_br]
        paths = [_Path(self.N, self.n, llr_internal)]

        for l in [bit_reversed(i, self.n) for i in range(self.N)]:
            new_paths = []
            for path in paths:
                self._update_llrs(path, l)
                llr_bit = path.L[l, self.n]
                if l in self.frozen_set:
                    pen = 0.0 if llr_bit >= 0 else abs(llr_bit)
                    path.pm += pen
                    path.B[l, self.n] = 0
                    self._update_bits(path, l)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        p2 = _Path(self.N, self.n, path.L[:, 0])
                        p2.L = path.L.copy()
                        p2.B = path.B.copy()
                        p2.pm = path.pm
                        p2.B[l, self.n] = bit
                        pen = 0.0 if (bit == 0 and llr_bit >= 0) or (bit == 1 and llr_bit < 0) else abs(llr_bit)
                        p2.pm += pen
                        self._update_bits(p2, l)
                        new_paths.append(p2)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        best = paths[0]
        if self.crc_length > 0:
            info_idx = np.array(sorted(set(range(self.N)) - self.frozen_set), dtype=int)
            valid = []
            for p in paths:
                u = p.B[:, self.n].astype(np.int8)
                if crc_check(u[info_idx], self.crc_length):
                    valid.append(p)
            if valid:
                best = min(valid, key=lambda p: p.pm)

        u_hat = best.B[:, self.n].astype(np.int8)
        return u_hat, best.pm

    def _update_llrs(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = int(path.B[j - branch_size, s + 1])
                    path.L[j, s + 1] = lower_llr(path.L[j, s], path.L[j - branch_size, s], top_bit)

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    path.B[j, s - 1] = path.B[j, s]
