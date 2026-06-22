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
    _f_boxplus,
    _g_boxplus,
    _prepare_channel_llr,
)
from encoder import bit_reversal_permutation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        reg = ((reg << 1) | int(bit)) & mask
        if reg & (1 << (crc_length - 1)):
            reg ^= poly
    for _ in range(crc_length):
        reg = (reg << 1) & mask
        if reg & (1 << (crc_length - 1)):
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    data = bits[:-crc_length]
    expected = crc_encode(data, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class _PathState:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        new = _PathState.__new__(_PathState)
        new.L = self.L.copy()
        new.B = self.B.copy()
        new.pm = self.pm
        new.u_hat = self.u_hat.copy()
        return new


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _propagate_llr(self, path, l):
        n, N = self.n, self.N
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _f_boxplus(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
                else:
                    path.L[j, s + 1] = _g_boxplus(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _propagate_bits(self, path, l):
        n, N = self.n, self.N
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        llr = _prepare_channel_llr(llr_ch)
        n, N = self.n, self.N
        paths = [_PathState(N, n)]
        paths[0].L[:, 0] = llr

        decode_order = [_bit_reversed(i, n) for i in range(N)]

        for l in decode_order:
            new_paths = []
            for path in paths:
                self._propagate_llr(path, l)
                llr_val = path.L[l, n]

                if l in self.frozen_set:
                    path.pm += self._pm_penalty(llr_val, 0)
                    path.u_hat[l] = 0
                    path.B[l, n] = 0
                    self._propagate_bits(path, l)
                    new_paths.append(path)
                else:
                    for u_bit in (0, 1):
                        child = path.copy()
                        child.pm += self._pm_penalty(llr_val, u_bit)
                        child.u_hat[l] = u_bit
                        child.B[l, n] = u_bit
                        self._propagate_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            best = min(valid or paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
