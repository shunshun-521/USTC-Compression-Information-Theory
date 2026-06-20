"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from channel import hard_decision_llr
from encoder import polar_encode


def _f_minsum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e7

    def decode(self, llr_ch):
        """主译码函数。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_bits] = self.large

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for layer in range(n - 1, -1, -1):
                stride = 1 << (n - 1 - layer)
                for block in range(0, N, 2 * stride):
                    for j in range(stride):
                        a = block + j
                        b = block + j + stride
                        L[layer, a] = _f_minsum(
                            L[layer + 1, a], L[layer + 1, b] + R[layer, b], self.alpha
                        )
                        L[layer, b] = _f_minsum(R[layer, a], L[layer + 1, a], self.alpha) + L[
                            layer + 1, b
                        ]

            for layer in range(0, n):
                stride = 1 << (n - 1 - layer)
                for block in range(0, N, 2 * stride):
                    for j in range(stride):
                        a = block + j
                        b = block + j + stride
                        R[layer + 1, a] = _f_minsum(
                            R[layer, b] + L[layer + 1, b], R[layer, a], self.alpha
                        )
                        R[layer + 1, b] = _f_minsum(
                            R[layer, a], L[layer + 1, a], self.alpha
                        ) + R[layer, b]

            u_hat = self._hard_decision(L, R)
            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters

    def _hard_decision(self, L, R):
        u_hat = np.zeros(self.N, dtype=np.int8)
        total = L[0, :] + R[0, :]
        for i in range(self.N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total[i] >= 0 else 1
        return u_hat
