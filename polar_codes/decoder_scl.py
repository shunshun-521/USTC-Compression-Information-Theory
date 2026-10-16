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
    _bit_reversed,
    _lower_llr,
    _upper_llr,
    sc_decode,
)
from encoder import bit_reversal_permutation
from utils import crc_check, crc_encode


class SCLDecoder:
    """SCL 译码器（Vangala 置换 SC 的列表扩展）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _pm_penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def _update_llrs(self, l, L, B):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, l, B):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        br = bit_reversal_permutation(self.N)
        llr0 = np.asarray(llr_ch, dtype=np.float64)[br]

        paths = []
        L0 = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B0 = np.full((self.N, self.n + 1), np.nan)
        L0[:, 0] = llr0
        paths.append({"L": L0, "B": B0, "pm": 0.0})

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            new_paths = []
            for path in paths:
                L, B, pm = path["L"], path["B"], path["pm"]
                self._update_llrs(l, L, B)
                cur_llr = L[l, self.n]

                if l in self.frozen_set:
                    pen = self._pm_penalty(cur_llr, 0)
                    B[l, self.n] = 0
                    self._update_bits(l, B)
                    new_paths.append({"L": L, "B": B, "pm": pm + pen})
                else:
                    for u in (0, 1):
                        cp = copy.deepcopy(path)
                        cL, cB = cp["L"], cp["B"]
                        pen = self._pm_penalty(cur_llr, u)
                        cB[l, self.n] = u
                        self._update_bits(l, cB)
                        cp["pm"] += pen
                        new_paths.append(cp)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        best = paths[0]
        u_hat = best["B"][:, self.n].astype(int)

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p["B"][:, self.n].astype(int)[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                best = min(valid, key=lambda p: p["pm"])
                u_hat = best["B"][:, self.n].astype(int)

        return u_hat, best["pm"]


__all__ = ["SCLDecoder", "crc_encode", "crc_check"]
