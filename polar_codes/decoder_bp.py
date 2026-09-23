"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _f_minsum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.brp = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        channel = llr_ch[self.brp]

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = channel
        R[:, 0] = 0.0
        frozen_idx = np.where(self.frozen_bits == 1)[0]
        R[frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, step * 2):
                    for k in range(step):
                        idx_u = i + k
                        idx_l = i + k + step
                        L[idx_u, j - 1] = _f_minsum(
                            R[idx_u, j] + L[idx_l, j], L[idx_u, j], self.alpha
                        )
                        L[idx_l, j - 1] = _f_minsum(
                            R[idx_u, j], L[idx_u, j], self.alpha
                        ) + L[idx_l, j]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, step * 2):
                    for k in range(step):
                        idx_u = i + k
                        idx_l = i + k + step
                        R[idx_u, j + 1] = _f_minsum(
                            R[idx_l, j] + L[idx_l, j + 1], R[idx_u, j], self.alpha
                        )
                        R[idx_l, j + 1] = _f_minsum(
                            R[idx_u, j], L[idx_u, j + 1], self.alpha
                        ) + R[idx_l, j]

            num_iters = it
            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits == 1] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (channel < 0).astype(int)
            x_hat_br = x_hat[self.brp]
            if np.array_equal(x_hat_br, hard_ch):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits == 1] = 0
        return u_hat, num_iters
