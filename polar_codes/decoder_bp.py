"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _ms_f(a, b, alpha):
    """min-sum f 运算，带修正因子 alpha"""
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        # 列 0..n：0 为信源端，n 为信道端
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            # 从右到左更新 L（列 n-1 到 0）
            for j in range(n - 1, -1, -1):
                stride = 2 ** j
                for i in range(0, N, 2 * stride):
                    for k in range(stride):
                        Li = i + k
                        Lis = i + k + stride
                        L[Li, j] = _ms_f(
                            L[Li, j + 1],
                            L[Lis, j + 1] + R[Lis, j],
                            self.alpha,
                        )
                        L[Lis, j] = _ms_f(R[Li, j], L[Li, j + 1], self.alpha) + L[
                            Lis, j + 1
                        ]

            # 从左到右更新 R（列 0 到 n-1）
            for j in range(0, n):
                stride = 2 ** j
                for i in range(0, N, 2 * stride):
                    for k in range(stride):
                        Li = i + k
                        Lis = i + k + stride
                        R[Li, j + 1] = _ms_f(
                            R[Lis, j + 1] + L[Lis, j + 1],
                            R[Li, j],
                            self.alpha,
                        )
                        R[Lis, j + 1] = _ms_f(R[Li, j], L[Li, j + 1], self.alpha) + R[
                            Lis, j
                        ]

            total_llr = L[:, 0] + R[:, 0]
            u_hat = (total_llr < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total_llr = L[:, 0] + R[:, 0]
        u_hat = (total_llr < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
