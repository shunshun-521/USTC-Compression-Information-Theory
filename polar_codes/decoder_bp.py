"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _boxplus_min_sum(a, b, alpha=0.9375):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, num_iters。"""
        llr = np.asarray(llr_ch, dtype=np.float64)[self.br]
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it

            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for block in range(0, N, 2 * s):
                    for k in range(s):
                        i = block + k
                        L[i, j - 1] = _boxplus_min_sum(
                            R[i, j] + L[i + s, j], L[i, j], self.alpha
                        )
                        L[i + s, j - 1] = (
                            _boxplus_min_sum(R[i, j], L[i, j], self.alpha) + L[i + s, j]
                        )

            for j in range(0, n):
                s = 1 << j
                for block in range(0, N, 2 * s):
                    for k in range(s):
                        i = block + k
                        R[i, j + 1] = _boxplus_min_sum(
                            R[i + s, j] + L[i + s, j + 1], R[i, j], self.alpha
                        )
                        R[i + s, j + 1] = (
                            _boxplus_min_sum(R[i, j], L[i, j + 1], self.alpha) + R[i + s, j]
                        )

            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[total < 0] = 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[total < 0] = 1
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
