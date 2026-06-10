"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _f_min_sum(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6

    def _hard_decision(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(np.int8)
        u_hat[self.frozen_bits] = 0
        return u_hat

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self._large

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it

            # 从右到左更新 L 消息
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_u = i + k
                        idx_v = i + k + s
                        L[idx_u, j - 1] = _f_min_sum(
                            R[idx_u, j] + L[idx_v, j],
                            L[idx_u, j],
                            self.alpha,
                        )
                        L[idx_v, j - 1] = _f_min_sum(
                            R[idx_u, j],
                            L[idx_u, j],
                            self.alpha,
                        ) + L[idx_v, j]

            # 从左到右更新 R 消息
            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_u = i + k
                        idx_v = i + k + s
                        R[idx_u, j] = _f_min_sum(
                            R[idx_v, j] + L[idx_v, j],
                            R[idx_u, j - 1],
                            self.alpha,
                        )
                        R[idx_v, j] = _f_min_sum(
                            R[idx_u, j - 1],
                            L[idx_u, j],
                            self.alpha,
                        ) + R[idx_v, j]

            u_hat = self._hard_decision(L, R)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard_ch):
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters
