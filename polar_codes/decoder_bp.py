"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def _hard_bits(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it

            for j in range(n, 0, -1):
                stride = 2 ** (j - 1)
                for i in range(0, N, 2 * stride):
                    for k in range(stride):
                        idx = i + k
                        s = idx + stride
                        L[idx, j - 1] = _f_min_sum(
                            R[idx, j] + L[s, j], L[idx, j], self.alpha
                        )
                        L[s, j - 1] = _f_min_sum(
                            R[idx, j], L[idx, j], self.alpha
                        ) + L[s, j]

            for j in range(0, n):
                stride = 2 ** j
                for i in range(0, N, 2 * stride):
                    for k in range(stride):
                        idx = i + k
                        s = idx + stride
                        R[idx, j + 1] = _f_min_sum(
                            R[s, j] + L[s, j + 1], R[idx, j], self.alpha
                        )
                        R[s, j + 1] = _f_min_sum(
                            R[idx, j], L[idx, j + 1], self.alpha
                        ) + R[s, j]

            u_hat = self._hard_bits(L, R)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        u_hat = self._hard_bits(L, R)
        return u_hat.astype(int), num_iters
