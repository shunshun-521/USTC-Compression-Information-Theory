"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    active_bit_level,
    active_llr_level,
    bit_reversed_index,
    f_operation,
    g_operation,
)
from encoder import bit_reversal_permutation

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | int(bit)) & mask
        if msb ^ int(bit):
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
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, L, B, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, B, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    @staticmethod
    def _metric_penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        paths = [{
            "pm": 0.0,
            "L": np.zeros((N, n + 1), dtype=np.float64),
            "B": np.zeros((N, n + 1), dtype=np.int8),
            "u_hat": np.zeros(N, dtype=int),
        }]
        paths[0]["L"][:, 0] = llr_ch[bit_reversal_permutation(N)]

        for phi in range(N):
            l = bit_reversed_index(phi, n)
            new_paths = []

            for path in paths:
                L = path["L"]
                B = path["B"]
                self._update_llrs(L, B, l)
                llr = L[l, n]

                if self.frozen_bits[l]:
                    child = {
                        "pm": path["pm"] + (abs(llr) if llr < 0 else 0.0),
                        "L": L.copy(),
                        "B": B.copy(),
                        "u_hat": path["u_hat"].copy(),
                    }
                    child["u_hat"][l] = 0
                    child["B"][l, n] = 0
                    self._update_bits(child["B"], l)
                    new_paths.append(child)
                else:
                    for bit in (0, 1):
                        child = {
                            "pm": path["pm"] + self._metric_penalty(llr, bit),
                            "L": L.copy(),
                            "B": B.copy(),
                            "u_hat": path["u_hat"].copy(),
                        }
                        child["u_hat"][l] = bit
                        child["B"][l, n] = bit
                        self._update_bits(child["B"], l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        best = min(paths, key=lambda p: p["pm"])
        if self.crc_length > 0:
            for path in sorted(paths, key=lambda p: p["pm"]):
                if crc_check(path["u_hat"][self.info_indices], self.crc_length):
                    best = path
                    break

        return best["u_hat"].copy(), best["pm"]
