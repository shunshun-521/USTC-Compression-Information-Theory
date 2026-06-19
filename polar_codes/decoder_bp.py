"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import f_operation, reorder_channel_llr
from encoder import polar_encode


class BPDecoder:
    """
    BP 译码器。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def _hard_decision(self, L, R):
        total = L + R
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat

    def decode(self, llr_ch):
        """
        主译码函数。
        """
        llr_ch = reorder_channel_llr(llr_ch)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it

            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, step * 2):
                    i2 = i + step
                    L[i, j - 1] = self._f_min_sum(
                        R[i, j] + L[i2, j], L[i, j]
                    )
                    L[i2, j - 1] = self._f_min_sum(R[i, j], L[i, j]) + L[i2, j]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, step * 2):
                    i2 = i + step
                    R[i, j + 1] = self._f_min_sum(
                        R[i2, j] + L[i2, j + 1], R[i, j]
                    )
                    R[i2, j + 1] = self._f_min_sum(R[i, j], L[i, j + 1]) + R[i2, j]

            u_hat = self._hard_decision(L[:, 0], R[:, 0])
            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                break

        u_hat = self._hard_decision(L[:, 0], R[:, 0])
        return u_hat, num_iters
