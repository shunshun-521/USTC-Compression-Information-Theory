"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    sign_a = np.where(a >= 0, 1.0, -1.0)
    sign_b = np.where(b >= 0, 1.0, -1.0)
    return alpha * sign_a * sign_b * np.minimum(np.abs(a), np.abs(b))


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
        """主译码函数。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, 0] = llr_ch
        R[:, n] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    L[i : i + s, j + 1] = _f_min_sum(
                        R[i : i + s, j] + L[i + s : i + 2 * s, j],
                        L[i : i + s, j],
                        self.alpha,
                    )
                    L[i + s : i + 2 * s, j + 1] = (
                        _f_min_sum(R[i : i + s, j], L[i : i + s, j], self.alpha)
                        + L[i + s : i + 2 * s, j]
                    )

            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    R[i : i + s, j - 1] = _f_min_sum(
                        R[i + s : i + 2 * s, j] + L[i + s : i + 2 * s, j],
                        R[i : i + s, j],
                        self.alpha,
                    )
                    R[i + s : i + 2 * s, j - 1] = (
                        _f_min_sum(R[i : i + s, j], L[i : i + s, j], self.alpha)
                        + R[i + s : i + 2 * s, j]
                    )

            total = L[:, n] + R[:, n]
            u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[:, n] + R[:, n]
        u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(int)
        u_hat[self.frozen_bits] = 0

        return u_hat, num_iters
