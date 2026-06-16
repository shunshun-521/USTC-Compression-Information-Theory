"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import f_operation
from encoder import bit_reversal_permutation, polar_encode


class BPDecoder:
    """BP 译码器（因子图 min-sum，含早停）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        br = bit_reversal_permutation(N)
        self.inv_br = np.argsort(br)
        self.large = 1e6

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def _hard_bits(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, num_iters"""
        llr = np.asarray(llr_ch, dtype=np.float64)[self.inv_br]
        N, n = self.N, self.n
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, 0] = llr
        R[:, n] = 0.0
        R[self.frozen_bits, n] = self.large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for s in range(n):
                block = 1 << (s + 1)
                half = 1 << s
                for j in range(0, N, block):
                    for k in range(half):
                        idx = j + k
                        L[idx, s + 1] = self._f_ms(
                            R[idx, s] + L[idx + half, s], L[idx, s]
                        )
                        L[idx + half, s + 1] = self._f_ms(
                            R[idx, s], L[idx, s]
                        ) + L[idx + half, s]

            for s in range(n - 1, -1, -1):
                block = 1 << (s + 1)
                half = 1 << s
                for j in range(0, N, block):
                    for k in range(half):
                        idx = j + k
                        R[idx, s] = self._f_ms(
                            R[idx + half, s + 1] + L[idx + half, s + 1], R[idx, s + 1]
                        )
                        R[idx + half, s] = self._f_ms(
                            R[idx, s + 1], L[idx, s + 1]
                        ) + R[idx + half, s + 1]

            u_hat = self._hard_bits(L, R)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        u_hat = self._hard_bits(L, R)
        return u_hat, num_iters
