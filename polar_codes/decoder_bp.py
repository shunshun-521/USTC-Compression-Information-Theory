"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _minsum_f(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for stage in range(n - 1, -1, -1):
                step = 1 << stage
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx = i + k
                        L[idx, stage] = _minsum_f(
                            R[idx, stage + 1] + L[idx + step, stage + 1],
                            L[idx, stage + 1],
                            alpha,
                        )
                        L[idx + step, stage] = _minsum_f(
                            R[idx, stage + 1],
                            L[idx, stage + 1],
                            alpha,
                        ) + L[idx + step, stage + 1]

            for stage in range(0, n):
                step = 1 << stage
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx = i + k
                        R[idx, stage + 1] = _minsum_f(
                            R[idx + step, stage] + L[idx + step, stage + 1],
                            R[idx, stage],
                            alpha,
                        )
                        R[idx + step, stage + 1] = _minsum_f(
                            R[idx, stage],
                            L[idx, stage + 1],
                            alpha,
                        ) + R[idx + step, stage]

            for i in range(N):
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        for i in range(N):
            u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
