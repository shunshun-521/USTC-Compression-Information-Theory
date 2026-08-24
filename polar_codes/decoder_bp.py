"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from channel import hard_decision_llr
from encoder import polar_encode


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

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        alpha = self.alpha

        # 列 0..n 为因子图，列 n 连接信道
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0

        for it in range(1, self.max_iter + 1):
            # 右到左更新 L（列 n -> 1）
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    L[i:i + s, j - 1] = _f_min_sum(
                        R[i:i + s, j] + L[i + s:i + 2 * s, j],
                        L[i:i + s, j],
                        alpha,
                    )
                    L[i + s:i + 2 * s, j - 1] = _f_min_sum(
                        R[i:i + s, j],
                        L[i:i + s, j],
                        alpha,
                    ) + L[i + s:i + 2 * s, j]

            # 左到右更新 R（列 0 -> n-1）
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    R[i:i + s, j + 1] = _f_min_sum(
                        R[i + s:i + 2 * s, j] + L[i + s:i + 2 * s, j + 1],
                        R[i:i + s, j],
                        alpha,
                    )
                    R[i + s:i + 2 * s, j + 1] = _f_min_sum(
                        R[i:i + s, j],
                        L[i:i + s, j + 1],
                        alpha,
                    ) + R[i + s:i + 2 * s, j]

            num_iters = it
            total = L[:, 0] + R[:, 0]
            u_hat = np.where(total >= 0, 0, 1).astype(np.int8)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_decision_llr(llr_ch)):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.where(total >= 0, 0, 1).astype(np.int8)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
