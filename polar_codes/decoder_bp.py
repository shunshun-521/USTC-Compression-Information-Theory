"""
极化码 BP（置信传播）译码器
基于因子图，min-sum 近似，含早停
"""
import numpy as np
import math
from encoder import polar_encode
from channel import hard_decision_llr


def _minsum_f(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

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
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                rcol = min(j + 1, n)
                for i in range(0, N, 2 * s):
                    L[i, j - 1] = _minsum_f(
                        R[i, j] + L[i + s, j], L[i, rcol], self.alpha
                    )
                    L[i + s, j - 1] = _minsum_f(
                        R[i, j], L[i, rcol], self.alpha
                    ) + L[i + s, rcol]

            for j in range(1, n + 1):
                s = 1 << (j - 1)
                rcol = min(j + 1, n)
                for i in range(0, N, 2 * s):
                    R[i, j] = _minsum_f(
                        R[i + s, j] + L[i + s, rcol], R[i, j - 1], self.alpha
                    )
                    R[i + s, j] = _minsum_f(
                        R[i, j - 1], L[i, rcol], self.alpha
                    ) + R[i + s, j - 1]

            for i in range(N):
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

        for i in range(N):
            u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
        u_hat[self.frozen_bits] = 0

        return u_hat.astype(int), num_iters
