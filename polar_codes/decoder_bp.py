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

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def _hard_bits(self, llr_ch):
        return (llr_ch < 0).astype(int)

    def _check_early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        x_hard = self._hard_bits(llr_ch)
        return np.array_equal(x_hat, x_hard)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx_l = i + k
                        idx_r = i + k + step
                        L[idx_l, j - 1] = _f_min_sum(
                            R[idx_l, j] + L[idx_r, j + 1],
                            L[idx_l, j + 1],
                            alpha,
                        )
                        L[idx_r, j - 1] = _f_min_sum(
                            R[idx_l, j], L[idx_l, j + 1], alpha
                        ) + L[idx_r, j + 1]

            for j in range(1, n + 1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx_l = i + k
                        idx_r = i + k + step
                        R[idx_l, j - 1] = _f_min_sum(
                            R[idx_r, j] + L[idx_r, j + 1],
                            R[idx_l, j - 1],
                            alpha,
                        )
                        R[idx_r, j - 1] = _f_min_sum(
                            R[idx_l, j - 1], L[idx_l, j + 1], alpha
                        ) + R[idx_r, j]

            for i in range(N):
                u_hat[i] = 0 if self.frozen_bits[i] else (
                    0 if (L[i, 0] + R[i, 0]) >= 0 else 1
                )

            num_iters = it
            if self._check_early_stop(u_hat, llr_ch):
                break

        for i in range(N):
            u_hat[i] = 0 if self.frozen_bits[i] else (
                0 if (L[i, 0] + R[i, 0]) >= 0 else 1
            )

        return u_hat, num_iters
