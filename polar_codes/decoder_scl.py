"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import os
import sys

import numpy as np

from decoder_sc import sc_decode

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """信息比特后附加 CRC 校验位。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        reg = 0
        for b in info_bits:
            reg ^= int(b) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) & 0xFF) ^ CRC8_POLY
                else:
                    reg = (reg << 1) & 0xFF
        crc_bits = np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=np.int8)
    elif crc_length == 16:
        reg = 0
        for b in info_bits:
            reg ^= int(b) << 15
            for _ in range(16):
                if reg & 0x8000:
                    reg = ((reg << 1) & 0xFFFF) ^ CRC16_POLY
                else:
                    reg = (reg << 1) & 0xFFFF
        crc_bits = np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=np.int8)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


def _import_scd():
    from decoder_sc import _sc_ref_imports

    with _sc_ref_imports():
        from decoder_utils import (
            active_bit_level,
            active_llr_level,
            lower_llr,
            upper_llr,
        )
        from polar_sc_utils import bit_reversed

    return bit_reversed, active_llr_level, active_bit_level, upper_llr, lower_llr


class _Path:
    __slots__ = ("L", "B", "pm", "N", "n")

    def __init__(self, N, n, llr_ch):
        self.N = N
        self.n = n
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch

    def clone(self):
        p = _Path(self.N, self.n, self.L[:, 0])
        p.pm = self.pm
        p.L = self.L.copy()
        p.B = self.B.copy()
        return p


class SCLDecoder:
    """SCL 译码器（路径级 Lazy Copy）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

        (
            self._bit_reversed,
            self._active_llr_level,
            self._active_bit_level,
            self._upper_llr,
            self._lower_llr,
        ) = _import_scd()

    def _update_llrs(self, path, l):
        for s in range(self.n - self._active_llr_level(l, self.n), self.n):
            block_size = int(2 ** (s + 1))
            branch_size = int(block_size / 2)
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = self._upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = self._lower_llr(
                        path.L[j - branch_size, s], path.L[j, s], path.B[j - branch_size, s + 1]
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - self._active_bit_level(l, self.n), -1):
            block_size = int(2 ** s)
            branch_size = int(block_size / 2)
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _metric_penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = self._bit_reversed(i, self.n)
            candidates = []
            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]
                if l in self.frozen_set:
                    opts = [0]
                else:
                    opts = [0, 1]
                for bit in opts:
                    child = path.clone()
                    child.pm += self._metric_penalty(llr, bit)
                    child.B[l, self.n] = bit
                    self._update_bits(child, l)
                    candidates.append(child)
            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        u_hat = paths[0].B[:, self.n].astype(int)

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.B[:, self.n].astype(int)[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.B[:, self.n].astype(int), best.pm
