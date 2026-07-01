"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self._large = 1e6

    def _f_min_sum(self, x, y):
        return self.alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)"""
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self._large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            # 右到左更新 L
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_u = i + k
                        idx_l = i + k + s
                        L[idx_u, j - 1] = self._f_min_sum(
                            R[idx_u, j - 1] + L[idx_l, j],
                            L[idx_u, j],
                        )
                        L[idx_l, j - 1] = self._f_min_sum(
                            R[idx_u, j - 1],
                            L[idx_u, j],
                        ) + L[idx_l, j]

            # 左到右更新 R
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_u = i + k
                        idx_l = i + k + s
                        R[idx_u, j + 1] = self._f_min_sum(
                            R[idx_l, j] + L[idx_l, j + 1],
                            R[idx_u, j],
                        )
                        R[idx_l, j + 1] = self._f_min_sum(
                            R[idx_u, j],
                            L[idx_u, j + 1],
                        ) + R[idx_l, j]

            u_hat = self._hard_decision(L, R)
            if self._early_stop(u_hat, llr_ch):
                num_iters = it
                break
        else:
            u_hat = self._hard_decision(L, R)

        return u_hat, num_iters

    def _hard_decision(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat

    def _early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, hard_ch)
