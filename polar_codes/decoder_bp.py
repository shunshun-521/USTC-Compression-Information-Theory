"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """
    BP 译码器。
    因子图列 0..n：列 0 为信源端，列 n 为信道端。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.large = 1e6

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_idx] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for phi in range(n):
                step = 1 << phi
                for beta in range(0, N, 2 * step):
                    for omega in range(step):
                        idx = beta + omega
                        idx2 = idx + step
                        R[phi + 1, idx] = self._f_min_sum(
                            R[phi, idx], L[phi + 1, idx2] + R[phi, idx2]
                        )
                        R[phi + 1, idx2] = self._f_min_sum(
                            R[phi, idx], L[phi + 1, idx2]
                        ) + R[phi, idx2]

            for phi in range(n - 1, -1, -1):
                step = 1 << phi
                for beta in range(0, N, 2 * step):
                    for omega in range(step):
                        idx = beta + omega
                        idx2 = idx + step
                        L[phi, idx] = self._f_min_sum(
                            L[phi + 1, idx], L[phi + 1, idx2] + R[phi, idx2]
                        )
                        L[phi, idx2] = self._f_min_sum(
                            R[phi, idx], L[phi + 1, idx2]
                        ) + L[phi + 1, idx2]

            for i in range(N):
                total = L[0, i] + R[0, i]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            total = L[0, i] + R[0, i]
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
