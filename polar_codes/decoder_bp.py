"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _f_min_sum(a, b, alpha):
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
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_orig = llr_ch.copy()
        br = bit_reversal_permutation(self.N)
        llr_ch = llr_ch[br]

        n = self.n
        N = self.N

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_idx] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for phi in range(n - 1, -1, -1):
                step = 1 << phi
                for beta in range(0, N, 2 * step):
                    for omega in range(step):
                        i = beta + omega
                        j = i + step
                        L[phi, i] = _f_min_sum(
                            R[phi + 1, i] + L[phi + 1, j], L[phi + 1, i], self.alpha
                        )
                        L[phi, j] = (
                            _f_min_sum(R[phi + 1, i], L[phi + 1, i], self.alpha)
                            + L[phi + 1, j]
                        )

            for phi in range(n):
                step = 1 << phi
                for beta in range(0, N, 2 * step):
                    for omega in range(step):
                        i = beta + omega
                        j = i + step
                        R[phi + 1, i] = _f_min_sum(
                            R[phi + 1, j] + L[phi + 1, j], R[phi, i], self.alpha
                        )
                        R[phi + 1, j] = (
                            _f_min_sum(R[phi, i], L[phi + 1, i], self.alpha)
                            + R[phi + 1, j]
                        )

            post = L[0, :] + R[0, :]
            u_hat = (post < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_orig < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        post = L[0, :] + R[0, :]
        u_hat = (post < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
