"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    """

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        # L[i][j]: 从右到左的消息; R[i][j]: 从左到右的消息
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            # 从右到左更新 L
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for block in range(0, N, 2 * s):
                    for i in range(block, block + s):
                        L[i, j - 1] = _f_min_sum(
                            R[i, j] + L[i + s, j], L[i, j], self.alpha
                        )
                        L[i + s, j - 1] = _f_min_sum(R[i, j], L[i, j], self.alpha) + L[
                            i + s, j
                        ]

            # 从左到右更新 R
            for j in range(0, n):
                s = 1 << j
                for block in range(0, N, 2 * s):
                    for i in range(block, block + s):
                        R[i, j + 1] = _f_min_sum(
                            R[i + s, j] + L[i + s, j + 1], R[i, j], self.alpha
                        )
                        R[i + s, j + 1] = _f_min_sum(
                            R[i, j], L[i, j + 1], self.alpha
                        ) + R[i + s, j]

            # 早停检查
            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[total < 0] = 1
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[total < 0] = 1
        u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
