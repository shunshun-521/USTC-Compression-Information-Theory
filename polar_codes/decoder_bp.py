"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _boxplus_minsum(a, b, alpha):
    sa, sb = np.sign(a), np.sign(b)
    return alpha * sa * sb * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def _iterate(self, L, R, llr_ch):
        n, N = self.n, self.N
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        for j in range(n - 1, -1, -1):
            step = 1 << j
            for i in range(0, N, 2 * step):
                L[i, j] = _boxplus_minsum(
                    R[i, j + 1] + L[i + step, j + 1], L[i, j + 1], self.alpha
                )
                L[i + step, j] = _boxplus_minsum(
                    R[i, j + 1], L[i, j + 1], self.alpha
                ) + L[i + step, j + 1]

        for j in range(0, n):
            step = 1 << j
            for i in range(0, N, 2 * step):
                R[i, j + 1] = _boxplus_minsum(
                    R[i + step, j] + L[i + step, j + 1], R[i, j], self.alpha
                )
                R[i + step, j + 1] = _boxplus_minsum(
                    R[i, j], L[i, j + 1], self.alpha
                ) + R[i + step, j]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            self._iterate(L, R, llr_ch)

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        return u_hat, num_iters
