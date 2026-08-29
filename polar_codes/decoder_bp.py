"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _f_min_sum(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器（分层因子图 min-sum）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.LARGE = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for lam in range(n - 1, -1, -1):
                step = 1 << lam
                for phi in range(0, N, 2 * step):
                    for omega in range(step):
                        i1 = phi + omega
                        i2 = phi + omega + step
                        L[i1, lam] = _f_min_sum(
                            L[i1, lam + 1] + R[i1, lam], L[i2, lam + 1], alpha
                        )
                        L[i2, lam] = _f_min_sum(R[i1, lam], L[i1, lam + 1], alpha) + L[i2, lam + 1]

            for lam in range(1, n + 1):
                step = 1 << (lam - 1)
                for phi in range(0, N, 2 * step):
                    for omega in range(step):
                        i1 = phi + omega
                        i2 = phi + omega + step
                        R[i1, lam] = _f_min_sum(
                            R[i2, lam] + L[i2, lam], R[i1, lam - 1], alpha
                        )
                        R[i2, lam] = _f_min_sum(R[i1, lam - 1], L[i1, lam], alpha) + R[i2, lam - 1]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(np.int8)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(np.int8)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
