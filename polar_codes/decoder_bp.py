"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _boxplus_minsum(a, b, alpha=1.0):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（Arikan 核 [[1,1],[0,1]] 因子图）。"""

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

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for block in range(0, N, 2 * s):
                    for i in range(block, block + s):
                        L[i, j - 1] = _boxplus_minsum(
                            R[i, j] + L[i + s, j], L[i, j], self.alpha
                        )
                        L[i + s, j - 1] = _boxplus_minsum(
                            R[i, j], L[i, j], self.alpha
                        ) + L[i + s, j]

            for j in range(0, n):
                s = 1 << j
                for block in range(0, N, 2 * s):
                    for i in range(block, block + s):
                        R[i, j + 1] = _boxplus_minsum(
                            R[i + s, j] + L[i + s, j + 1], R[i, j], self.alpha
                        )
                        R[i + s, j + 1] = _boxplus_minsum(
                            R[i, j], L[i, j + 1], self.alpha
                        ) + R[i + s, j]

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard):
                num_iters = it
                break
            num_iters = it

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
