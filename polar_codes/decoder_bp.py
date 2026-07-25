"""
极化码 BP（置信传播）译码器
基于因子图 min-sum BP（与 polar_encode 配套），含早停
"""
import math

import numpy as np

from encoder import polar_encode


def _g_minsum(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e8

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N, a = self.n, self.N, self.alpha

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_idx] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            L[n, :] = llr_ch

            for i in range(n - 1, -1, -1):
                step = 1 << i
                for j in range(0, N, step * 2):
                    L[i, j] = _g_minsum(
                        L[i + 1, j], L[i + 1, j + step] + R[i, j + step], a
                    )
                    L[i, j + step] = _g_minsum(
                        L[i + 1, j], R[i, j], a
                    ) + L[i + 1, j + step]

            for i in range(0, n):
                step = 1 << i
                for j in range(0, N, step * 2):
                    R[i + 1, j] = _g_minsum(
                        R[i, j], L[i + 1, j + step] + R[i, j + step], a
                    )
                    R[i + 1, j + step] = _g_minsum(
                        L[i + 1, j], R[i, j], a
                    ) + R[i, j + step]

            for j in range(N):
                u_hat[j] = 0 if (L[0, j] + R[0, j]) >= 0 else 1
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, (llr_ch < 0).astype(int)):
                break

        for j in range(N):
            u_hat[j] = 0 if (L[0, j] + R[0, j]) >= 0 else 1
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
