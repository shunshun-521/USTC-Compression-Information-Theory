"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _f_min_sum(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """
    BP 译码器。
    """

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self.frozen_internal = self.frozen_bits[self.br].astype(bool)
        self.frozen_idx_internal = np.where(self.frozen_internal)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx_internal, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n - 1, -1, -1):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx_u = i + k
                        idx_l = i + k + step
                        L[idx_u, j] = _f_min_sum(
                            R[idx_u, j] + L[idx_l, j + 1], L[idx_u, j + 1], self.alpha
                        )
                        L[idx_l, j] = _f_min_sum(
                            R[idx_u, j], L[idx_u, j + 1], self.alpha
                        ) + L[idx_l, j + 1]

            for j in range(n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx_u = i + k
                        idx_l = i + k + step
                        R[idx_u, j + 1] = _f_min_sum(
                            R[idx_l, j] + L[idx_l, j + 1], R[idx_u, j], self.alpha
                        )
                        R[idx_l, j + 1] = (
                            _f_min_sum(R[idx_u, j], L[idx_u, j + 1], self.alpha)
                            + R[idx_l, j]
                        )

            num_iters = it
            for i in range(N):
                if self.frozen_internal[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            u_out = np.zeros(N, dtype=int)
            u_out[self.br] = u_hat
            x_hat = polar_encode(u_out)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        u_internal = np.zeros(N, dtype=int)
        for i in range(N):
            if self.frozen_internal[i]:
                u_internal[i] = 0
            else:
                u_internal[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        u_hat = np.zeros(N, dtype=int)
        u_hat[self.br] = u_internal
        return u_hat, num_iters
