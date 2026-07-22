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
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_bits] = self.large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for stage in range(n - 1, -1, -1):
                step = 1 << stage
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        a = i + k
                        b = i + k + step
                        L[stage, a] = _minsum_f(
                            R[stage + 1, a] + L[stage + 1, b],
                            L[stage + 1, a],
                            self.alpha,
                        )
                        L[stage, b] = _minsum_f(
                            R[stage + 1, a],
                            L[stage + 1, a],
                            self.alpha,
                        ) + L[stage + 1, b]

            for stage in range(1, n + 1):
                step = 1 << (stage - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        a = i + k
                        b = i + k + step
                        R[stage, a] = _minsum_f(
                            R[stage - 1, b] + L[stage, b],
                            R[stage - 1, a],
                            self.alpha,
                        )
                        R[stage, b] = _minsum_f(
                            R[stage - 1, a],
                            L[stage, a],
                            self.alpha,
                        ) + R[stage - 1, b]

            total = L[0, :] + R[0, :]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[0, :] + R[0, :]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
