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
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self.br_inv = np.argsort(self.br)

    def decode(self, llr_ch):
        """返回 (u_hat, num_iters)"""
        N, n = self.N, self.n
        llr_nat = np.asarray(llr_ch, dtype=np.float64)
        llr = llr_nat[self.br].copy()

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.br_inv[self.frozen_bits], 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                jc = min(j + 1, n)
                for i in range(0, N, 2 * s):
                    L[i, j - 1] = _f_min_sum(
                        R[i, j] + L[i + s, jc], L[i, jc], self.alpha
                    )
                    L[i + s, j - 1] = (
                        _f_min_sum(R[i, j], L[i, jc], self.alpha) + L[i + s, jc]
                    )

            for j in range(0, n):
                s = 1 << j
                jc = min(j + 1, n)
                for i in range(0, N, 2 * s):
                    R[i, j + 1] = _f_min_sum(
                        R[i + s, j] + L[i + s, jc], R[i, j], self.alpha
                    )
                    R[i + s, j + 1] = (
                        _f_min_sum(R[i, j], L[i, jc], self.alpha) + R[i + s, j]
                    )

            for i in range(N):
                nat_i = self.br_inv[i]
                total = L[i, 0] + R[i, 0]
                if self.frozen_bits[nat_i]:
                    u_hat[nat_i] = 0
                else:
                    u_hat[nat_i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard = (llr_nat < 0).astype(int)
            if np.array_equal(x_hat, hard):
                num_iters = it
                break

        for i in range(N):
            nat_i = self.br_inv[i]
            total = L[i, 0] + R[i, 0]
            if self.frozen_bits[nat_i]:
                u_hat[nat_i] = 0
            else:
                u_hat[nat_i] = 0 if total >= 0 else 1

        return u_hat, num_iters
