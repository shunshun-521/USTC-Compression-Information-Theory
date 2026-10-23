"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _f(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, 0] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            for s in range(n):
                block = 2 ** (s + 1)
                half = block // 2
                for base in range(0, N, block):
                    for j in range(half):
                        i = base + j
                        L[i, s + 1] = self._f(R[i, s] + L[i + half, s], L[i, s])
                        L[i + half, s + 1] = self._f(R[i, s], L[i, s]) + L[i + half, s]

            for s in range(n - 1, -1, -1):
                block = 2 ** (s + 1)
                half = block // 2
                for base in range(0, N, block):
                    for j in range(half):
                        i = base + j
                        R[i, s + 1] = self._f(
                            R[i + half, s] + L[i + half, s + 1], R[i, s]
                        )
                        R[i + half, s + 1] = self._f(R[i, s], L[i, s + 1]) + R[i + half, s]

            total = L[:, n] + R[:, n]
            for i in range(N):
                u_hat[i] = 0 if self.frozen_bits[i] or total[i] >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                break

        total = L[:, n] + R[:, n]
        for i in range(N):
            u_hat[i] = 0 if self.frozen_bits[i] or total[i] >= 0 else 1

        return u_hat, num_iters
