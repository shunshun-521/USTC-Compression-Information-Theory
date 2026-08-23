"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from channel import hard_decision_llr
from encoder import polar_encode


def ms_f(a, b, alpha):
    """Min-sum f operation with scaling factor alpha."""
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（因子图 min-sum，含早停）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(self.max_iter):
            num_iters = it + 1

            for j in range(n, 0, -1):
                s = 2 ** (j - 1)
                for block in range(2 ** (n - j)):
                    base = block * 2 * s
                    for i in range(s):
                        idx_u = base + i
                        idx_l = base + s + i
                        L[idx_u, j - 1] = ms_f(
                            R[idx_u, j] + L[idx_l, j],
                            L[idx_u, j],
                            alpha,
                        )
                        L[idx_l, j - 1] = ms_f(
                            R[idx_u, j],
                            L[idx_u, j],
                            alpha,
                        ) + L[idx_l, j]

            for j in range(n):
                s = 2 ** j
                for block in range(2 ** (n - j - 1)):
                    base = block * 2 * s
                    for i in range(s):
                        idx_u = base + i
                        idx_l = base + s + i
                        R[idx_u, j + 1] = ms_f(
                            R[idx_l, j] + L[idx_l, j + 1],
                            R[idx_u, j],
                            alpha,
                        )
                        R[idx_l, j + 1] = ms_f(
                            R[idx_u, j],
                            L[idx_u, j + 1],
                            alpha,
                        ) + R[idx_l, j]

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_decision_llr(llr_ch)):
                break

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat.astype(int), num_iters
