"""
极化码 SCL（串行抵消列表）译码器，含 CRC 辅助 CA-SCL
"""
import copy

import numpy as np

from decoder_sc import (
    _decoder_llr_order,
    active_bit_level,
    active_llr_level,
    bit_reversed,
    lower_llr,
    upper_llr,
)
from utils import crc_check, crc_encode


class SCLDecoder:
    """SCL 译码（基于置换 SC 树；路径复制为浅拷贝 + 数组复用）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits.astype(bool))[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _new_path(self, llr):
        return {
            "L": np.full((self.N, self.n + 1), np.nan, dtype=np.float64),
            "B": np.zeros((self.N, self.n + 1), dtype=np.int8),
            "pm": 0.0,
            "u": np.zeros(self.N, dtype=int),
        }

    def _update_llrs(self, path, l):
        L = path["L"]
        B = path["B"]
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = lower_llr(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        L = path["L"]
        B = path["B"]
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr = _decoder_llr_order(np.asarray(llr_ch, dtype=np.float64), self.N)
        paths = [self._new_path(llr)]
        paths[0]["L"][:, 0] = llr

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            new_paths = []
            for path in paths:
                self._update_llrs(path, l)
                llr_bit = path["L"][l, self.n]
                if l in self.frozen_set:
                    u0 = 0
                    pm = path["pm"] + (0.0 if llr_bit >= 0 else abs(llr_bit))
                    p0 = copy.deepcopy(path)
                    p0["pm"] = pm
                    p0["u"][l] = u0
                    p0["B"][l, self.n] = u0
                    self._update_bits(p0, l)
                    new_paths.append(p0)
                else:
                    for u_bit in (0, 1):
                        pm = path["pm"] + (
                            0.0 if (llr_bit >= 0 and u_bit == 0) or (llr_bit < 0 and u_bit == 1) else abs(llr_bit)
                        )
                        p = copy.deepcopy(path)
                        p["pm"] = pm
                        p["u"][l] = u_bit
                        p["B"][l, self.n] = u_bit
                        self._update_bits(p, l)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        best = min(paths, key=lambda p: p["pm"])
        if self.crc_length > 0:
            info_idx = np.where(self.frozen_bits == 0)[0]
            info_bits = best["u"][info_idx]
            valid = [p for p in paths if crc_check(info_bits, self.crc_length)]
            if valid:
                best = min(valid, key=lambda p: p["pm"])
        return best["u"], best["pm"]
