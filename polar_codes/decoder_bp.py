"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode
from decoder_sc import f_operation


def _f_min_sum(a, b, alpha):
    return alpha * f_operation(a, b)


class BPDecoder:
    """BP 译码器（按层更新 L/R 消息）"""

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

        # L[s, i], R[s, i]: s=0..n，s=n 为信道侧
        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_bits] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # 右到左更新 L
            for s in range(n, 0, -1):
                step = 1 << (s - 1)
                for i in range(0, N, step * 2):
                    for t in range(step):
                        Li = i + t
                        Li2 = i + t + step
                        L[s - 1, Li] = _f_min_sum(
                            R[s - 1, Li] + L[s, Li2], L[s, Li], self.alpha
                        )
                        L[s - 1, Li2] = _f_min_sum(
                            R[s - 1, Li], L[s, Li], self.alpha
                        ) + L[s, Li2]

            # 左到右更新 R
            for s in range(0, n):
                step = 1 << s
                for i in range(0, N, step * 2):
                    for t in range(step):
                        Li = i + t
                        Li2 = i + t + step
                        R[s + 1, Li] = _f_min_sum(
                            R[s + 1, Li2] + L[s + 1, Li2], R[s, Li], self.alpha
                        )
                        R[s + 1, Li2] = _f_min_sum(
                            R[s, Li], L[s + 1, Li], self.alpha
                        ) + R[s + 1, Li2]

            num_iters = it

            for i in range(N):
                u_hat[i] = 0 if self.frozen_bits[i] else (0 if (L[0, i] + R[0, i]) >= 0 else 1)

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                break

        for i in range(N):
            u_hat[i] = 0 if self.frozen_bits[i] else (0 if (L[0, i] + R[0, i]) >= 0 else 1)

        return u_hat, num_iters
