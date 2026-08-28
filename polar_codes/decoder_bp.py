"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode

LARGE = 1e6


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            for s in range(n - 1, -1, -1):
                step = 1 << s
                for i in range(0, N, step * 2):
                    for j in range(i, i + step):
                        L[j, s] = self._f_min_sum(
                            R[j, s] + L[j + step, s + 1], L[j, s + 1]
                        )
                        L[j + step, s] = (
                            self._f_min_sum(R[j, s], L[j, s + 1]) + L[j + step, s + 1]
                        )

            for s in range(n):
                step = 1 << s
                for i in range(0, N, step * 2):
                    for j in range(i, i + step):
                        R[j, s + 1] = self._f_min_sum(
                            R[j + step, s] + L[j + step, s + 1], R[j, s]
                        )
                        R[j + step, s + 1] = (
                            self._f_min_sum(R[j, s], L[j, s + 1]) + R[j + step, s]
                        )

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
