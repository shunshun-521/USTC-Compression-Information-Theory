"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _minsum_f(x, y, alpha):
    """min-sum f 运算，带归一化因子 alpha"""
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），列 n 为信道接收端。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            # 从右到左更新 L 消息
            for lam in range(n - 1, -1, -1):
                step = 1 << lam
                for phi in range(0, N, step << 1):
                    for beta in range(step):
                        i = phi + beta
                        j = i + step
                        L[i, lam] = _minsum_f(
                            R[i, lam + 1] + L[j, lam + 1],
                            L[i, lam + 1],
                            self.alpha,
                        )
                        L[j, lam] = _minsum_f(
                            R[i, lam + 1],
                            L[i, lam + 1],
                            self.alpha,
                        ) + L[j, lam + 1]

            # 从左到右更新 R 消息
            for lam in range(0, n):
                step = 1 << lam
                for phi in range(0, N, step << 1):
                    for beta in range(step):
                        i = phi + beta
                        j = i + step
                        R[i, lam + 1] = _minsum_f(
                            R[j, lam] + L[j, lam + 1],
                            R[i, lam],
                            self.alpha,
                        )
                        R[j, lam + 1] = _minsum_f(
                            R[i, lam],
                            L[i, lam + 1],
                            self.alpha,
                        ) + R[j, lam]

            num_iters = it
            total_llr = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=np.int8)
            u_hat[~self.frozen_bits] = (total_llr[~self.frozen_bits] < 0).astype(np.int8)
            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, x_hard):
                break

        total_llr = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=np.int8)
        u_hat[~self.frozen_bits] = (total_llr[~self.frozen_bits] < 0).astype(np.int8)
        return u_hat, num_iters
