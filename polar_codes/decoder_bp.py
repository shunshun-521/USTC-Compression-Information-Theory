"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _f_minsum(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


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
        n = self.n
        N = self.N
        llr = llr_ch[self.br].astype(np.float64)

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # 从右到左更新 L（列 n-1 到 0）
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for block in range(0, N, 2 * s):
                    for i in range(block, block + s):
                        L[i, j] = _f_minsum(
                            R[i, j + 1] + L[i + s, j + 1], L[i, j + 1], self.alpha
                        )
                        L[i + s, j] = _f_minsum(
                            R[i, j + 1], L[i, j + 1], self.alpha
                        ) + L[i + s, j + 1]

            # 从左到右更新 R（列 0 到 n-1）
            for j in range(0, n):
                s = 1 << j
                for block in range(0, N, 2 * s):
                    for i in range(block, block + s):
                        R[i, j + 1] = _f_minsum(
                            R[i + s, j] + L[i + s, j + 1], R[i, j], self.alpha
                        )
                        R[i + s, j + 1] = (
                            _f_minsum(R[i, j], L[i, j + 1], self.alpha) + R[i + s, j]
                        )

            num_iters = it

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    total = L[i, 0] + R[i, 0]
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
