"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import bit_reversed_index, f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]
        self.LARGE = 1e6

    def _f_min_sum(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, 0] = llr_ch
        R[:, n] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it
            for s in range(n - 1, -1, -1):
                block = 1 << (s + 1)
                half = block // 2
                for start in range(0, N, block):
                    for j in range(start, start + half):
                        L[j, s + 1] = self._f_min_sum(
                            R[j, s] + L[j + half, s],
                            L[j, s],
                        )
                        L[j + half, s + 1] = self._f_min_sum(R[j, s], L[j, s]) + L[j + half, s]

            for s in range(0, n):
                block = 1 << (s + 1)
                half = block // 2
                for start in range(0, N, block):
                    for j in range(start, start + half):
                        R[j, s + 1] = self._f_min_sum(
                            R[j + half, s] + L[j + half, s + 1],
                            R[j, s],
                        )
                        R[j + half, s + 1] = self._f_min_sum(R[j, s], L[j, s + 1]) + R[j + half, s]

            total = L[:, n] + R[:, n]
            u_hat = np.zeros(N, dtype=int)
            u_hat[self.info_idx] = (total[self.info_idx] < 0).astype(int)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        total = L[:, n] + R[:, n]
        u_hat = np.zeros(N, dtype=int)
        u_hat[self.info_idx] = (total[self.info_idx] < 0).astype(int)
        return u_hat, num_iters
