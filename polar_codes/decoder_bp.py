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
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def _hard_decision(self, L, R):
        total = L + R
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits == 1] = 0
        return u_hat

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        N, n = self.N, self.n
        llr = np.asarray(llr_ch, dtype=np.float64)

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr
        R[:, 0] = 0.0
        frozen_idx = np.where(self.frozen_bits == 1)[0]
        R[frozen_idx, 0] = LARGE

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx = i + k
                        s = idx + step
                        L[idx, j - 1] = self._f_min_sum(
                            R[idx, j] + L[s, j], L[idx, j]
                        )
                        L[s, j - 1] = self._f_min_sum(R[idx, j], L[idx, j]) + L[s, j]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx = i + k
                        s = idx + step
                        R[idx, j + 1] = self._f_min_sum(
                            R[s, j] + L[s, j + 1], R[idx, j]
                        )
                        R[s, j + 1] = self._f_min_sum(R[idx, j], L[idx, j + 1]) + R[
                            s, j
                        ]

            u_hat = self._hard_decision(L[:, 0], R[:, 0])
            x_hat = polar_encode(u_hat)
            hard_ch = (llr < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
            num_iters = it

        u_hat = self._hard_decision(L[:, 0], R[:, 0])
        return u_hat, num_iters
