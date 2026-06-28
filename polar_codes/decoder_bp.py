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
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        fb = np.asarray(frozen_bits)
        if fb.dtype == bool:
            self.frozen_bits = fb
        else:
            self.frozen_bits = fb == 1
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        m = n + 1

        L = np.zeros((N, m), dtype=np.float64)
        R = np.zeros((N, m), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n - 1, -1, -1):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    L[i, j] = _minsum_f(
                        R[i, j + 1] + L[i + step, j + 1],
                        L[i, j + 1],
                        self.alpha,
                    )
                    L[i + step, j] = _minsum_f(
                        R[i, j + 1], L[i, j + 1], self.alpha
                    ) + L[i + step, j + 1]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    R[i, j + 1] = _minsum_f(
                        R[i + step, j] + L[i + step, j + 1],
                        R[i, j],
                        self.alpha,
                    )
                    R[i + step, j + 1] = _minsum_f(
                        R[i, j], L[i, j + 1], self.alpha
                    ) + R[i + step, j]

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1
                if self.frozen_bits[i]:
                    u_hat[i] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
            num_iters = it

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            u_hat[i] = 0 if total >= 0 else 1
            if self.frozen_bits[i]:
                u_hat[i] = 0

        return u_hat, num_iters
