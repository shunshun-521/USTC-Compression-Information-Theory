"""
极化码 SCL（串行抵消列表）译码器
基于 Permuted SCD 的 L/B 阵列，含 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    active_bit_level,
    active_llr_level,
    bit_reversed_index,
    lower_llr,
    upper_llr,
)
from encoder import bit_reversal_permutation
from utils import crc_encode_bits, crc_check_bits


def crc_encode(info_bits, crc_length=8):
    return crc_encode_bits(info_bits, crc_length)


def crc_check(bits, crc_length=8):
    return crc_check_bits(bits, crc_length)


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        fb = np.asarray(frozen_bits)
        self.frozen_bits = fb.astype(bool) if fb.dtype == bool else (fb != 0)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def _init_path(self, llr_ch):
        N, n = self.N, self.n
        L = np.full((N, n + 1), np.nan, dtype=np.float64)
        B = np.full((N, n + 1), np.nan)
        L[:, 0] = llr_ch
        return {"L": L, "B": B, "pm": 0.0, "u": np.zeros(N, dtype=int)}

    def _update_llrs(self, path, l):
        L, B = path["L"], path["B"]
        n, N = self.n, self.N
        for s in range(n - active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    if np.isnan(top_bit):
                        top_bit = 0
                    L[j, s + 1] = lower_llr(
                        L[j, s], L[j - branch_size, s], int(top_bit)
                    )

    def _update_bits(self, path, l):
        B = path["B"]
        n, N = self.n, self.N
        if l < N // 2:
            return
        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def _pm_penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.br]
        n, N = self.n, self.N
        paths = [self._init_path(llr_ch)]

        for i in range(N):
            l = bit_reversed_index(i, n)
            candidates = []
            for pidx, path in enumerate(paths):
                self._update_llrs(path, l)
                llr = path["L"][l, n]
                if l in self.frozen_set:
                    candidates.append((path["pm"] + self._pm_penalty(llr, 0), pidx, 0))
                else:
                    for u in (0, 1):
                        candidates.append((path["pm"] + self._pm_penalty(llr, u), pidx, u))

            candidates.sort(key=lambda x: x[0])
            new_paths = []
            for pm, pidx, u in candidates[: self.list_size]:
                parent = paths[pidx]
                child = {
                    "L": parent["L"].copy(),
                    "B": parent["B"].copy(),
                    "pm": pm,
                    "u": parent["u"].copy(),
                }
                child["B"][l, n] = u
                child["u"][l] = u
                self._update_bits(child, l)
                new_paths.append(child)
            paths = new_paths

        best_pm = float("inf")
        best_u = None
        crc_ok = []
        for path in paths:
            u_hat = path["u"].astype(int)
            if self.crc_length > 0:
                payload = u_hat[~self.frozen_bits]
                if crc_check(payload, self.crc_length):
                    crc_ok.append((path["pm"], u_hat))
            if path["pm"] < best_pm:
                best_pm = path["pm"]
                best_u = u_hat

        if crc_ok:
            crc_ok.sort(key=lambda x: x[0])
            return crc_ok[0][1], crc_ok[0][0]
        return best_u, best_pm
