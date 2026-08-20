"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import _permute_channel_llrs, f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _f_min_sum(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        hard_ch = (np.asarray(llr_ch, dtype=np.float64) < 0).astype(int)
        llr_ch = _permute_channel_llrs(llr_ch, self.N)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for stage in range(n - 1, -1, -1):
                step = 1 << stage
                for i in range(0, N, 2 * step):
                    left = i
                    right = i + step
                    L[left, stage] = self._f_min_sum(
                        R[left, stage] + L[right, stage + 1], L[left, stage + 1]
                    )
                    L[right, stage] = (
                        self._f_min_sum(R[left, stage], L[left, stage + 1])
                        + L[right, stage + 1]
                    )

            for stage in range(0, n):
                step = 1 << stage
                for i in range(0, N, 2 * step):
                    left = i
                    right = i + step
                    R[right, stage + 1] = (
                        self._f_min_sum(R[left, stage], L[left, stage + 1])
                        + R[right, stage]
                    )
                    R[left, stage + 1] = self._f_min_sum(
                        R[right, stage + 1] + L[right, stage + 1], R[left, stage]
                    )

            u_hat = np.zeros(N, dtype=int)
            total = L[:, 0] + R[:, 0]
            u_hat[total < 0] = 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        u_hat = np.zeros(N, dtype=int)
        total = L[:, 0] + R[:, 0]
        u_hat[total < 0] = 1
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
