"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（min-sum + 早停）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _f_ms(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_bits] = self.large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for lam in range(n, 0, -1):
                step = 1 << (lam - 1)
                for i in range(0, N, step * 2):
                    for j in range(i, i + step):
                        L[lam - 1, j] = self._f_ms(
                            R[lam, j] + L[lam, j + step], L[lam, j]
                        )
                        L[lam - 1, j + step] = (
                            self._f_ms(R[lam, j], L[lam, j]) + L[lam, j + step]
                        )

            for lam in range(0, n):
                step = 1 << lam
                for i in range(0, N, step * 2):
                    for j in range(i, i + step):
                        R[lam + 1, j] = self._f_ms(
                            R[lam + 1, j + step] + L[lam + 1, j + step], R[lam, j]
                        )
                        R[lam + 1, j + step] = (
                            self._f_ms(R[lam, j], L[lam + 1, j]) + R[lam + 1, j + step]
                        )

            total = L[0, :] + R[0, :]
            u_hat = (total < 0).astype(np.int32)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(np.int32)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                return u_hat, num_iters
            num_iters = it

        total = L[0, :] + R[0, :]
        u_hat = (total < 0).astype(np.int32)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
