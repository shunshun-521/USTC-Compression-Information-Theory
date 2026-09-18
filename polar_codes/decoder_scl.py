"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _lower_llr,
    _upper_llr,
)
from encoder import bit_reversed_index


def _crc_bitwise(bits, crc_length):
    if crc_length == 8:
        poly = 0x07
        mask = 0xFF
        shift = 7
    elif crc_length == 16:
        poly = 0x8005
        mask = 0xFFFF
        shift = 15
    else:
        raise ValueError("crc_length must be 8 or 16")

    crc = 0
    for bit in bits:
        feedback = ((crc >> shift) ^ int(bit)) & 1
        crc = ((crc << 1) & mask) ^ (poly * feedback)
    return crc


def crc_encode(info_bits, crc_length=8):
    """CRC 编码。r=8: CRC-8 (0x07), r=16: CRC-16 (0x8005)"""
    info_bits = np.asarray(info_bits, dtype=int)
    reg = _crc_bitwise(info_bits, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    return _crc_bitwise(bits, crc_length) == 0


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def _update_llrs(self, L, B, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s],
                        L[j - branch_size, s],
                        B[j - branch_size, s + 1],
                    )

    def _update_bits(self, B, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [{
            "pm": 0.0,
            "u_hat": np.zeros(N, dtype=int),
            "L": np.zeros((N, n + 1), dtype=np.float64),
            "B": np.zeros((N, n + 1), dtype=int),
        }]
        paths[0]["L"][:, 0] = llr_ch

        for i in range(N):
            l = bit_reversed_index(i, n)
            for path in paths:
                self._update_llrs(path["L"], path["B"], l)

            candidates = []
            if self.frozen_bits[l]:
                for path in paths:
                    llr_val = path["L"][l, n]
                    new_path = copy.deepcopy(path)
                    if llr_val < 0:
                        new_path["pm"] += abs(llr_val)
                    new_path["u_hat"][l] = 0
                    new_path["B"][l, n] = 0
                    self._update_bits(new_path["B"], l)
                    candidates.append(new_path)
            else:
                for path in paths:
                    llr_val = path["L"][l, n]
                    hard = 0 if llr_val >= 0 else 1
                    for u_val in (0, 1):
                        new_path = copy.deepcopy(path)
                        if u_val != hard:
                            new_path["pm"] += abs(llr_val)
                        new_path["u_hat"][l] = u_val
                        new_path["B"][l, n] = u_val
                        self._update_bits(new_path["B"], l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p["u_hat"][~self.frozen_bits], self.crc_length)
            ]
            best = min(valid if valid else paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["u_hat"].copy(), best["pm"]
