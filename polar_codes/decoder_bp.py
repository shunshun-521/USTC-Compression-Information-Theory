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
    """BP 译码器（分层因子图，列 0..n）"""

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

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            # L 消息：从右向左（列 n-1 -> 0）
            for j in range(n - 1, -1, -1):
                step = 1 << j
                for base in range(0, N, 2 * step):
                    for k in range(step):
                        i = base + k
                        ip = i + step
                        L[i, j] = _f_min_sum(
                            R[i, j + 1] + L[ip, j + 1], L[i, j + 1], self.alpha
                        )
                        L[ip, j] = (
                            _f_min_sum(R[i, j + 1], L[i, j + 1], self.alpha) + L[ip, j + 1]
                        )

            # R 消息：从左向右（列 0 -> n-1）
            for j in range(n):
                step = 1 << j
                for base in range(0, N, 2 * step):
                    for k in range(step):
                        i = base + k
                        ip = i + step
                        R[i, j + 1] = _f_min_sum(
                            R[ip, j + 1] + L[ip, j + 1], R[i, j], self.alpha
                        )
                        R[ip, j + 1] = (
                            _f_min_sum(R[i, j], L[i, j + 1], self.alpha) + R[ip, j + 1]
                        )

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
