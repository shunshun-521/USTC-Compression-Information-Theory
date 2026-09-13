"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from channel import hard_decision_llr
from encoder import polar_encode


def _boxplus_minsum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_bits] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for s in range(n, 0, -1):
                step = 1 << (s - 1)
                for i in range(0, N, 2 * step):
                    for j in range(i, i + step):
                        L[s - 1, j] = _boxplus_minsum(
                            R[s, j] + L[s, j + step], L[s, j], self.alpha
                        )
                        L[s - 1, j + step] = _boxplus_minsum(
                            R[s, j], L[s, j], self.alpha
                        ) + L[s, j + step]

            for s in range(n):
                step = 1 << s
                for i in range(0, N, 2 * step):
                    for j in range(i, i + step):
                        R[s + 1, j] = _boxplus_minsum(
                            R[s + 1, j + step] + L[s + 1, j + step],
                            R[s, j],
                            self.alpha,
                        )
                        R[s + 1, j + step] = _boxplus_minsum(
                            R[s, j], L[s + 1, j], self.alpha
                        ) + R[s + 1, j + step]

            total = L[0, :] + R[0, :]
            u_hat = (total < 0).astype(np.int8)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_decision_llr(llr_ch)):
                break

        total = L[0, :] + R[0, :]
        u_hat = (total < 0).astype(np.int8)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
