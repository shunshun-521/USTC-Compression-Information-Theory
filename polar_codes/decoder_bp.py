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
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def decode(self, llr_ch):
        """返回 (u_hat, num_iters)"""
        N = self.N
        n = self.n
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        frozen_mask = self.frozen_bits.astype(bool)

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, frozen_mask] = self.large

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for stage in range(n - 1, -1, -1):
                step = 1 << stage
                for i in range(0, N, 2 * step):
                    L[stage, i:i + step] = _f_min_sum(
                        R[stage, i:i + step] + L[stage + 1, i + step:i + 2 * step],
                        L[stage + 1, i:i + step],
                        self.alpha,
                    )
                    L[stage, i + step:i + 2 * step] = (
                        _f_min_sum(
                            R[stage, i:i + step],
                            L[stage + 1, i:i + step],
                            self.alpha,
                        )
                        + L[stage + 1, i + step:i + 2 * step]
                    )

            for stage in range(0, n):
                step = 1 << stage
                for i in range(0, N, 2 * step):
                    R[stage + 1, i:i + step] = _f_min_sum(
                        R[stage, i + step:i + 2 * step] + L[stage + 1, i + step:i + 2 * step],
                        R[stage, i:i + step],
                        self.alpha,
                    )
                    R[stage + 1, i + step:i + 2 * step] = (
                        _f_min_sum(
                            R[stage, i:i + step],
                            L[stage + 1, i:i + step],
                            self.alpha,
                        )
                        + R[stage, i + step:i + 2 * step]
                    )

            total = L[0, :] + R[0, :]
            u_hat = (total < 0).astype(int)
            u_hat[frozen_mask] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[0, :] + R[0, :]
        u_hat = (total < 0).astype(int)
        u_hat[frozen_mask] = 0
        return u_hat, num_iters
