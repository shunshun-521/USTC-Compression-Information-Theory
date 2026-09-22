"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode
from channel import hard_decision_llr


def bp_f_min_sum(a, b, alpha=0.9375):
    """BP 因子图中的 min-sum f 运算。"""
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    Li, Li_s = i, i + s
                    L[Li, j - 1] = bp_f_min_sum(
                        R[Li, j - 1] + L[Li_s, j],
                        L[Li, j],
                        self.alpha,
                    )
                    L[Li_s, j - 1] = bp_f_min_sum(
                        R[Li, j - 1],
                        L[Li, j],
                        self.alpha,
                    ) + L[Li_s, j]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    Li, Li_s = i, i + s
                    R[Li, j + 1] = bp_f_min_sum(
                        R[Li_s, j] + L[Li_s, j + 1],
                        R[Li, j],
                        self.alpha,
                    )
                    R[Li_s, j + 1] = bp_f_min_sum(
                        R[Li, j],
                        L[Li, j + 1],
                        self.alpha,
                    ) + R[Li_s, j]

            u_hat = np.zeros(N, dtype=np.int8)
            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

        u_hat = np.zeros(N, dtype=np.int8)
        for i in range(N):
            total = L[i, 0] + R[i, 0]
            u_hat[i] = 0 if total >= 0 else 1
        u_hat[self.frozen_bits] = 0

        return u_hat, num_iters
