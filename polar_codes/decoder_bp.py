"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import f_operation
from encoder import bit_reversal_permutation, polar_encode


class BPDecoder:
    """BP 译码器（因子图列 0..n）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits).astype(int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.brp = bit_reversal_permutation(N)
        self.large = 1e6

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch[self.brp]
        R[:, 0] = 0.0
        for idx in np.where(self.frozen_bits == 1)[0]:
            R[idx, 0] = self.large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for t in range(step):
                        u = i + t
                        v = u + step
                        L[u, j - 1] = self._f_min_sum(
                            R[u, j] + L[v, j], L[u, j]
                        )
                        L[v, j - 1] = self._f_min_sum(R[u, j], L[u, j]) + L[v, j]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    for t in range(step):
                        u = i + t
                        v = u + step
                        R[u, j + 1] = self._f_min_sum(
                            R[v, j] + L[v, j + 1], R[u, j]
                        )
                        R[v, j + 1] = self._f_min_sum(R[u, j], L[u, j + 1]) + R[v, j]

            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                if self.frozen_bits[i] == 1:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break

        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            if self.frozen_bits[i] == 1:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
        return u_hat, num_iters
