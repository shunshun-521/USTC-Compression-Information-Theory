"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _boxplus_min_sum(a, b, alpha=0.9375):
    """min-sum 近似的 box-plus 运算"""
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """
    BP 译码器。
    """

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def decode(self, llr_ch):
        """
        主译码函数。
        """
        N, n = self.N, self.n
        llr_ch = llr_ch.astype(np.float64)

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                span = 1 << (j - 1)
                for i in range(0, N, 2 * span):
                    for k in range(span):
                        idx = i + k
                        idx2 = idx + span
                        L[idx, j - 1] = _boxplus_min_sum(
                            R[idx, j] + L[idx2, j], L[idx, j], self.alpha
                        )
                        L[idx2, j - 1] = _boxplus_min_sum(
                            R[idx, j], L[idx, j], self.alpha
                        ) + L[idx2, j]

            for j in range(0, n):
                span = 1 << j
                for i in range(0, N, 2 * span):
                    for k in range(span):
                        idx = i + k
                        idx2 = idx + span
                        R[idx, j + 1] = _boxplus_min_sum(
                            R[idx2, j] + L[idx2, j + 1], R[idx, j], self.alpha
                        )
                        R[idx2, j + 1] = _boxplus_min_sum(
                            R[idx, j], L[idx2, j + 1], self.alpha
                        ) + R[idx2, j]

            for i in range(N):
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
            num_iters = it

        for i in range(N):
            u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
        u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
