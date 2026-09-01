"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _ms_f(x, y, alpha):
    """min-sum f 运算（带修正因子 alpha）"""
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


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
        n = self.n
        N = self.N
        alpha = self.alpha

        # L[i][j]: 从右到左消息; R[i][j]: 从左到右消息; j=0..n
        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # 从右到左更新 L（列 n-1 到 0）
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        L[i + k, j] = _ms_f(
                            R[i + k, j] + L[i + k + s, j + 1],
                            L[i + k, j + 1], alpha,
                        )
                        L[i + k + s, j] = _ms_f(
                            R[i + k, j], L[i + k, j + 1], alpha,
                        ) + L[i + k + s, j + 1]

            # 从左到右更新 R（列 0 到 n-1）
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        R[i + k, j + 1] = _ms_f(
                            R[i + k + s, j] + L[i + k + s, j + 1],
                            R[i + k, j], alpha,
                        )
                        R[i + k + s, j + 1] = _ms_f(
                            R[i + k, j], L[i + k, j + 1], alpha,
                        ) + R[i + k + s, j]

            for i in range(N):
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        return u_hat, num_iters
