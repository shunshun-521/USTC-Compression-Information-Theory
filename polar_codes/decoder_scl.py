"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_utils_internal import (
    active_bit_level,
    active_llr_level,
    hard_decision,
    lower_llr,
    upper_llr,
)
from encoder import bit_reversal_permutation
from polar_sc_utils import bit_reversed
from utils import crc_check, crc_encode


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, likelihoods):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = likelihoods
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（路径复制采用延迟拷贝索引）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _update_llrs(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = lower_llr(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        int(path.B[j - branch_size, s + 1]),
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _path_metric_penalty(self, llr, bit):
        decided = 0 if llr >= 0 else 1
        return 0.0 if bit == decided else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        likelihoods = llr_ch[br]

        paths = [_Path(self.N, self.n, likelihoods.copy())]

        for phi in range(self.N):
            l = bit_reversed(phi, self.n)
            new_paths = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    path.u_hat[l] = 0
                    path.B[l, self.n] = 0
                    path.pm += self._path_metric_penalty(llr, 0)
                    self._update_bits(path, l)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        child = _Path(self.N, self.n, path.L[:, 0].copy())
                        child.L[:] = path.L
                        child.B[:] = path.B
                        child.u_hat[:] = path.u_hat
                        child.pm = path.pm + self._path_metric_penalty(llr, bit)
                        child.u_hat[l] = bit
                        child.B[l, self.n] = bit
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        best = paths[0]
        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            valid = [p for p in paths if crc_check(p.u_hat[info_idx], self.crc_length)]
            if valid:
                best = min(valid, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
