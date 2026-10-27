"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy

import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    bit_reversed_index,
    active_llr_level,
    active_bit_level,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < crc_length:
        return False
    return np.array_equal(bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length)[-crc_length:])


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _new_path(self, llr_ch):
        path = {
            "L": np.full((self.N, self.n + 1), np.nan, dtype=np.float64),
            "B": np.full((self.N, self.n + 1), np.nan, dtype=np.float64),
            "pm": 0.0,
            "u_hat": np.zeros(self.N, dtype=np.int8),
        }
        path["L"][:, 0] = llr_ch
        return path

    def _update_llrs(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path["L"][j, s + 1] = f_operation(
                        path["L"][j, s], path["L"][j + branch_size, s]
                    )
                else:
                    path["L"][j, s + 1] = g_operation(
                        path["L"][j - branch_size, s],
                        path["L"][j, s],
                        path["B"][j - branch_size, s + 1],
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path["B"][j - branch_size, s - 1] = int(path["B"][j, s]) ^ int(
                        path["B"][j - branch_size, s]
                    )
                    path["B"][j, s - 1] = path["B"][j, s]

    def _penalty(self, llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for phi in range(self.N):
            l = bit_reversed_index(phi, self.n)
            new_paths = []

            for path in paths:
                self._update_llrs(path, l)
                llr_leaf = path["L"][l, self.n]

                if l in self.frozen_set:
                    p = copy.deepcopy(path)
                    p["pm"] += self._penalty(llr_leaf, 0)
                    p["B"][l, self.n] = 0
                    p["u_hat"][l] = 0
                    self._update_bits(p, l)
                    new_paths.append(p)
                else:
                    for u_bit in (0, 1):
                        p = copy.deepcopy(path)
                        p["pm"] += self._penalty(llr_leaf, u_bit)
                        p["B"][l, self.n] = u_bit
                        p["u_hat"][l] = u_bit
                        self._update_bits(p, l)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p["u_hat"][self.info_indices], self.crc_length)
            ]
            best = min(valid if valid else paths, key=lambda p: p["pm"])
        else:
            best = paths[0]

        return best["u_hat"].copy(), best["pm"]
