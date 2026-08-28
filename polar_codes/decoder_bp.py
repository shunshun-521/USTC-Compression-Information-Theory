"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode
from channel import hard_decision_llr


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_min_sum(self, x, y):
        """min-sum f 运算，带修正因子 alpha"""
        return self.alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        n = self.n
        N = self.N
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        # L[i][j]: 从右到左的消息，R[i][j]: 从左到右的消息
        # 使用 (n+1) 列，列 0 为信源端，列 n 为信道端
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(self.max_iter):
            num_iters = it + 1

            # 从右到左更新 L 消息（列 n-1 到 0）
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_l = i + k
                        idx_r = i + k + s
                        L[idx_l, j] = self._f_min_sum(
                            R[idx_l, j] + L[idx_r, j + 1], L[idx_l, j + 1]
                        )
                        L[idx_r, j] = self._f_min_sum(
                            R[idx_l, j], L[idx_l, j + 1]
                        ) + L[idx_r, j + 1]

            # 从左到右更新 R 消息（列 0 到 n-1）
            for j in range(n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_l = i + k
                        idx_r = i + k + s
                        R[idx_l, j + 1] = self._f_min_sum(
                            R[idx_r, j + 1] + L[idx_r, j + 1], R[idx_l, j]
                        )
                        R[idx_r, j + 1] = self._f_min_sum(
                            R[idx_l, j], L[idx_l, j + 1]
                        ) + R[idx_r, j]

            # 判决与早停
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                break

        # 最终判决
        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
