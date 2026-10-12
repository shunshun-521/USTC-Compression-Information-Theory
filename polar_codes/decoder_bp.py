"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _minsum_f(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（分层 min-sum）。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)

        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_idx] = self.LARGE

        u_hat = np.zeros(N, dtype=int)
        num_iters = 0

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for lamb in range(n, 0, -1):
                stride = 2 ** (lamb - 1)
                for phi in range(0, N, 2 * stride):
                    for beta in range(stride):
                        i = phi + beta
                        j = i + stride
                        L[lamb - 1, i] = _minsum_f(
                            R[lamb, i] + L[lamb, j], L[lamb, i], self.alpha
                        )
                        L[lamb - 1, j] = _minsum_f(
                            R[lamb, i], L[lamb, i], self.alpha
                        ) + L[lamb, j]

            for lamb in range(0, n):
                stride = 2 ** lamb
                for phi in range(0, N, 2 * stride):
                    for beta in range(stride):
                        i = phi + beta
                        j = i + stride
                        R[lamb + 1, i] = _minsum_f(
                            R[lamb, j] + L[lamb + 1, j], R[lamb, i], self.alpha
                        )
                        R[lamb + 1, j] = _minsum_f(
                            R[lamb, i], L[lamb + 1, i], self.alpha
                        ) + R[lamb, j]

            for i in range(N):
                total = L[0, i] + R[0, i]
                u_hat[i] = 0 if total >= 0 else 1
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[0, i] + R[0, i]) >= 0 else 1

        return u_hat, num_iters
