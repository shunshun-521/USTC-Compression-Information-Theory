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
    """BP 译码器（与 encoder 的 Kronecker 因子图配套）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_idx] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, step << 1):
                    for k in range(step):
                        idx = i + k
                        idx2 = idx + step
                        L[j - 1, idx] = _f_min_sum(
                            R[j, idx] + L[j, idx2], L[j, idx], self.alpha
                        )
                        L[j - 1, idx2] = (
                            _f_min_sum(R[j, idx], L[j, idx], self.alpha) + L[j, idx2]
                        )

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, step << 1):
                    for k in range(step):
                        idx = i + k
                        idx2 = idx + step
                        R[j + 1, idx] = _f_min_sum(
                            R[j + 1, idx2] + L[j + 1, idx2], R[j, idx], self.alpha
                        )
                        R[j + 1, idx2] = (
                            _f_min_sum(R[j, idx], L[j + 1, idx], self.alpha) + R[j + 1, idx2]
                        )

            num_iters = it
            total = L[0, :] + R[0, :]
            u_hat = np.where(total >= 0, 0, 1).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        total = L[0, :] + R[0, :]
        u_hat = np.where(total >= 0, 0, 1).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
