"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器（因子图列 0..n）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self._large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            # 右 -> 左：层 j = n-1 .. 0
            for j in range(n - 1, -1, -1):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    L[i, j] = self._f_ms(
                        R[i, j + 1] + L[i + step, j + 1], L[i, j + 1]
                    )
                    L[i + step, j] = self._f_ms(R[i, j + 1], L[i, j + 1]) + L[
                        i + step, j + 1
                    ]

            # 左 -> 右：层 j = 0 .. n-1
            for j in range(n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    R[i, j + 1] = self._f_ms(
                        R[i + step, j + 1] + L[i + step, j + 1], R[i, j]
                    )
                    R[i + step, j + 1] = self._f_ms(R[i, j], L[i, j + 1]) + R[
                        i + step, j + 1
                    ]

            u_hat = self._hard_decision(L, R)
            if self._early_stop(u_hat, llr_ch):
                num_iters = it
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters

    def _hard_decision(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat

    def _early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, hard_ch)
