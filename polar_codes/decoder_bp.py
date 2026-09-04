"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _ms_f(a, b, alpha):
    """min-sum f 函数"""
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（基于极化码因子图的分层消息传递）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.LARGE = 1e6

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, m, alpha = self.N, self.n, self.alpha

        # L[i, s]: 右向左消息；R[i, s]: 左向右消息；s=0..m
        L = np.zeros((N, m + 1), dtype=np.float64)
        R = np.zeros((N, m + 1), dtype=np.float64)

        L[:, m] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # 右向左更新 L
            for s in range(m - 1, -1, -1):
                block = 1 << (s + 1)
                half = 1 << s
                for base in range(0, N, block):
                    for j in range(half):
                        i = base + j
                        ip = i + half
                        L[i, s] = _ms_f(
                            L[i, s + 1],
                            R[ip, s] + L[ip, s + 1],
                            alpha,
                        )
                        L[ip, s] = _ms_f(R[i, s], L[i, s + 1], alpha) + L[ip, s + 1]

            # 左向右更新 R
            for s in range(0, m):
                block = 1 << (s + 1)
                half = 1 << s
                for base in range(0, N, block):
                    for j in range(half):
                        i = base + j
                        ip = i + half
                        R[i, s + 1] = _ms_f(
                            R[ip, s] + L[ip, s + 1],
                            R[i, s],
                            alpha,
                        )
                        R[ip, s + 1] = _ms_f(R[i, s], L[i, s + 1], alpha) + R[ip, s]

            num_iters = it

            # 早停
            posterior = L[:, 0] + R[:, 0]
            u_hat = (posterior < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                break

        posterior = L[:, 0] + R[:, 0]
        u_hat = (posterior < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
