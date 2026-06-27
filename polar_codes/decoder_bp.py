"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _bp_f(x, y, alpha):
    sx = np.where(x >= 0, 1.0, -1.0)
    sy = np.where(y >= 0, 1.0, -1.0)
    return alpha * sx * sy * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        if 2 ** self.n != N:
            raise ValueError("N must be a power of 2")
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_indices = np.where(self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6

    def _hard_bits(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_indices] = 0
        return u_hat

    def _check_early_stop(self, llr_ch, u_hat):
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, hard_ch)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_indices, 0] = self._large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx_u = i + k
                        idx_l = i + k + step
                        L[idx_u, j - 1] = _bp_f(
                            R[idx_u, j] + L[idx_l, j],
                            L[idx_u, j],
                            self.alpha,
                        )
                        L[idx_l, j - 1] = _bp_f(
                            R[idx_u, j],
                            L[idx_u, j],
                            self.alpha,
                        ) + L[idx_l, j]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx_u = i + k
                        idx_l = i + k + step
                        R[idx_u, j + 1] = _bp_f(
                            R[idx_l, j] + L[idx_l, j + 1],
                            R[idx_u, j],
                            self.alpha,
                        )
                        R[idx_l, j + 1] = _bp_f(
                            R[idx_u, j],
                            L[idx_u, j + 1],
                            self.alpha,
                        ) + R[idx_l, j]

            u_hat = self._hard_bits(L, R)
            if self._check_early_stop(llr_ch, u_hat):
                num_iters = it
                break

        u_hat = self._hard_bits(L, R)
        return u_hat, num_iters
