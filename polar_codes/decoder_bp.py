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
    """BP 译码器（Arikan 极化因子图）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.LARGE = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n, alpha = self.N, self.n, self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for stage in range(n - 1, -1, -1):
                step = 1 << stage
                for i in range(0, N, 2 * step):
                    for j in range(i, i + step):
                        L[j, stage] = _f_min_sum(
                            R[j, stage + 1] + L[j + step, stage + 1],
                            L[j, stage + 1],
                            alpha,
                        )
                        L[j + step, stage] = _f_min_sum(
                            R[j, stage + 1],
                            L[j, stage + 1],
                            alpha,
                        ) + L[j + step, stage + 1]

            for stage in range(n):
                step = 1 << stage
                for i in range(0, N, 2 * step):
                    for j in range(i, i + step):
                        R[j, stage + 1] = _f_min_sum(
                            R[j + step, stage + 1] + L[j + step, stage + 1],
                            R[j, stage],
                            alpha,
                        )
                        R[j + step, stage + 1] = _f_min_sum(
                            R[j, stage],
                            L[j, stage + 1],
                            alpha,
                        ) + R[j + step, stage]

            num_iters = it
            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
