"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        N, n = self.N, self.n
        alpha = self.alpha
        channel = np.asarray(llr_ch, dtype=np.float64)[self.br]

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = channel
        R[0, :] = 0.0
        R[0, self.frozen_bits] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            for stage in range(n, 0, -1):
                s = 1 << (stage - 1)
                for i in range(0, N, 2 * s):
                    for j in range(s):
                        a = i + j
                        b = i + j + s
                        L[stage - 1, a] = _f_min_sum(
                            R[stage, a] + L[stage, b], L[stage, a], alpha
                        )
                        L[stage - 1, b] = _f_min_sum(
                            R[stage, a], L[stage, a], alpha
                        ) + L[stage, b]

            for stage in range(0, n):
                s = 1 << stage
                for i in range(0, N, 2 * s):
                    for j in range(s):
                        a = i + j
                        b = i + j + s
                        R[stage + 1, a] = _f_min_sum(
                            R[stage + 1, b] + L[stage + 1, b], R[stage, a], alpha
                        )
                        R[stage + 1, b] = _f_min_sum(
                            R[stage, a], L[stage + 1, a], alpha
                        ) + R[stage + 1, b]

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[0, i] + R[0, i]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (channel < 0).astype(np.int8)
            x_natural = x_hat[self.br]
            if np.array_equal(x_natural, hard_ch):
                num_iters = it
                break
            num_iters = it

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[0, i] + R[0, i]) >= 0 else 1

        return u_hat, num_iters
