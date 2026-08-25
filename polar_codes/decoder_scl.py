"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    active_bit_level,
    active_llr_level,
    bit_reversed,
    bit_reversal_permutation,
    lower_llr,
    upper_llr,
)
from utils import crc_check, crc_encode


class SCLDecoder:
    """
    SCL 译码器（含 Lazy Copy 优化）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.list_size = list_size
        self.crc_length = crc_length
        self.rev = bit_reversal_permutation(N)

        frozen_bits = np.asarray(frozen_bits)
        if frozen_bits.dtype != bool:
            self.frozen_set = set(np.where(frozen_bits.astype(bool))[0])
        else:
            self.frozen_set = set(np.where(frozen_bits)[0])

    def _init_paths(self, llr_ch):
        paths = [
            {
                "pm": 0.0,
                "L": np.full((self.N, self.n + 1), np.nan, dtype=np.float64),
                "B": np.full((self.N, self.n + 1), np.nan, dtype=np.float64),
            }
        ]
        paths[0]["L"][:, 0] = llr_ch
        return paths

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
                        L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1])
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        B = path["B"]
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.rev]
        paths = self._init_paths(llr_ch)

        for phi in range(self.N):
            l = bit_reversed(phi, self.n)
            new_paths = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path["L"][l, self.n]

                if l in self.frozen_set:
                    bit = 0
                    pm = path["pm"] + self._path_metric_penalty(llr, bit)
                    child = {
                        "pm": pm,
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                    }
                    child["B"][l, self.n] = bit
                    self._update_bits(child, l)
                    new_paths.append(child)
                else:
                    for bit in (0, 1):
                        pm = path["pm"] + self._path_metric_penalty(llr, bit)
                        child = {
                            "pm": pm,
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                        }
                        child["B"][l, self.n] = bit
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        candidates = []
        for path in paths:
            u_hat = np.nan_to_num(path["B"][:, self.n], nan=0.0).astype(int)
            pm = path["pm"]
            candidates.append((pm, u_hat))

        if self.crc_length > 0:
            info_mask = np.ones(self.N, dtype=bool)
            for idx in self.frozen_set:
                info_mask[idx] = False
            info_idx = np.where(info_mask)[0]
            valid = [
                (pm, u)
                for pm, u in candidates
                if crc_check(u[info_idx], self.crc_length)
            ]
            if valid:
                candidates = valid

        best_pm, u_hat = min(candidates, key=lambda x: x[0])
        return u_hat, best_pm
