"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _ms_g(x, y, alpha):
    """min-sum 近似 box-plus 运算。"""
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


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
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            L[:, n] = llr_ch

            for j in range(n - 1, -1, -1):
                span = 1 << j
                for i in range(0, N, 2 * span):
                    for t in range(span):
                        i0 = i + t
                        i1 = i0 + span
                        L[i0, j] = _ms_g(L[i0, j + 1], L[i1, j + 1] + R[i1, j], self.alpha)
                        L[i1, j] = _ms_g(R[i0, j], L[i0, j + 1], self.alpha) + L[i1, j + 1]

            for j in range(0, n):
                span = 1 << j
                for i in range(0, N, 2 * span):
                    for t in range(span):
                        i0 = i + t
                        i1 = i0 + span
                        R[i0, j + 1] = _ms_g(R[i0, j], L[i1, j + 1] + R[i1, j], self.alpha)
                        R[i1, j + 1] = _ms_g(R[i0, j], L[i0, j + 1], self.alpha) + R[i1, j]

            u_hat = (L[:, 0] < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        u_hat = (L[:, 0] < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
