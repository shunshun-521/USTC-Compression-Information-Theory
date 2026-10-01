"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _boxplus_minsum(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * min(abs(x), abs(y))


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n - 1, -1, -1):
                step = 1 << j
                for i in range(0, N, step << 1):
                    for t in range(step):
                        top = i + t
                        bot = top + step
                        L[top, j] = _boxplus_minsum(
                            R[top, j] + L[bot, j + 1],
                            L[top, j + 1],
                            self.alpha,
                        )
                        L[bot, j] = (
                            _boxplus_minsum(R[top, j], L[top, j + 1], self.alpha)
                            + L[bot, j + 1]
                        )

            for j in range(1, n):
                step = 1 << (j - 1)
                for i in range(0, N, step << 1):
                    for t in range(step):
                        top = i + t
                        bot = top + step
                        R[top, j] = _boxplus_minsum(
                            R[bot, j] + L[bot, j + 1],
                            R[top, j - 1],
                            self.alpha,
                        )
                        R[bot, j] = (
                            _boxplus_minsum(R[top, j - 1], L[top, j + 1], self.alpha)
                            + R[bot, j]
                        )

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            rev = bit_reversal_permutation(N)
            hard = (llr_ch[rev] < 0).astype(int)
            if np.array_equal(x_hat, hard):
                num_iters = it
                break
            num_iters = it

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            u_hat[i] = 0 if total >= 0 else 1
        u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
