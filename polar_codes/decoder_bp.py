"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from channel import hard_decision_llr
from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e7

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n] = llr_ch
        R[0, self.frozen_idx] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for s in range(n, 0, -1):
                stride = 1 << (s - 1)
                for i in range(0, N, stride * 2):
                    for j in range(i, i + stride):
                        L[s - 1, j] = _f_min_sum(
                            R[s, j] + L[s, j + stride], L[s, j], self.alpha
                        )
                        L[s - 1, j + stride] = (
                            _f_min_sum(R[s, j], L[s, j], self.alpha) + L[s, j + stride]
                        )

            for s in range(0, n):
                stride = 1 << s
                for i in range(0, N, stride * 2):
                    for j in range(i, i + stride):
                        R[s + 1, j] = _f_min_sum(
                            R[s + 1, j + stride] + L[s + 1, j + stride],
                            R[s, j],
                            self.alpha,
                        )
                        R[s + 1, j + stride] = (
                            _f_min_sum(R[s, j], L[s + 1, j], self.alpha)
                            + R[s + 1, j + stride]
                        )

            total = L[0] + R[0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(int)

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_decision_llr(llr_ch)):
                num_iters = it
                break

        total = L[0] + R[0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(int)
        return u_hat, num_iters
