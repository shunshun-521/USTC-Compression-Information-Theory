"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    sa = np.sign(a)
    sb = np.sign(b)
    sa[sa == 0] = 1
    sb[sb == 0] = 1
    return alpha * sa * sb * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for stage in range(n - 1, -1, -1):
                span = 1 << stage
                for block in range(0, N, span << 1):
                    left = slice(block, block + span)
                    right = slice(block + span, block + 2 * span)
                    L[left, stage] = _f_min_sum(
                        R[left, stage + 1] + L[right, stage + 1],
                        L[left, stage + 1],
                        self.alpha,
                    )
                    L[right, stage] = (
                        _f_min_sum(R[left, stage + 1], L[left, stage + 1], self.alpha)
                        + L[right, stage + 1]
                    )

            for stage in range(n):
                span = 1 << stage
                for block in range(0, N, span << 1):
                    left = slice(block, block + span)
                    right = slice(block + span, block + 2 * span)
                    R[left, stage + 1] = _f_min_sum(
                        R[right, stage + 1] + L[right, stage + 1],
                        R[left, stage + 1],
                        self.alpha,
                    )
                    R[right, stage + 1] = (
                        _f_min_sum(R[left, stage + 1], L[left, stage + 1], self.alpha)
                        + R[right, stage + 1]
                    )

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
