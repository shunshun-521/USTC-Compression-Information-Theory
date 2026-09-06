"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.large = 1e6

    def decode(self, llr_ch):
        N = self.N
        n = self.n
        alpha = self.alpha
        half = N // 2

        llr = np.asarray(llr_ch, dtype=np.float64)
        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr
        R[0, :] = 0.0
        R[0, self.frozen_idx] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for i in range(n - 1, -1, -1):
                for j in range(half):
                    L[i, j] = _f_min_sum(
                        L[i + 1, 2 * j],
                        L[i + 1, 2 * j + 1] + R[i, j + half],
                        alpha,
                    )
                    L[i, j + half] = (
                        _f_min_sum(R[i, j], L[i + 1, 2 * j], alpha)
                        + L[i + 1, 2 * j + 1]
                    )

            for i in range(n):
                for j in range(half):
                    R[i + 1, 2 * j] = _f_min_sum(
                        R[i, j],
                        L[i + 1, 2 * j + 1] + R[i, j + half],
                        alpha,
                    )
                    R[i + 1, 2 * j + 1] = (
                        _f_min_sum(R[i, j], L[i + 1, 2 * j], alpha) + R[i, j + half]
                    )

            num_iters = it
            total = L[0, :] + R[0, :]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        total = L[0, :] + R[0, :]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
