"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _ms_f(x, y, alpha):
    sx, sy = np.sign(x), np.sign(y)
    sx = np.where(sx == 0, 1, sx)
    sy = np.where(sy == 0, 1, sy)
    return alpha * sx * sy * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)

        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_bits] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for stage in range(n - 1, -1, -1):
                bs = 1 << stage
                for i in range(0, N, 2 * bs):
                    for j in range(bs):
                        a, b = i + j, i + j + bs
                        L[stage, a] = _ms_f(
                            R[stage, a] + L[stage + 1, a],
                            L[stage + 1, b],
                            self.alpha,
                        )
                        L[stage, b] = _ms_f(
                            R[stage, a], L[stage + 1, a], self.alpha
                        ) + L[stage + 1, b]

            for stage in range(0, n):
                bs = 1 << stage
                for i in range(0, N, 2 * bs):
                    for j in range(bs):
                        a, b = i + j, i + j + bs
                        R[stage + 1, a] = _ms_f(
                            R[stage + 1, b] + L[stage + 1, b],
                            R[stage, a],
                            self.alpha,
                        )
                        R[stage + 1, b] = _ms_f(
                            R[stage, a], L[stage + 1, a], self.alpha
                        ) + R[stage + 1, b]

            for i in range(N):
                u_hat[i] = 0 if self.frozen_bits[i] else (
                    0 if (L[0, i] + R[0, i]) >= 0 else 1
                )

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        for i in range(N):
            u_hat[i] = 0 if self.frozen_bits[i] else (
                0 if (L[0, i] + R[0, i]) >= 0 else 1
            )

        return u_hat, num_iters
