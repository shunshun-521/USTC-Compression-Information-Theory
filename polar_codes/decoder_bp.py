"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _f_min_sum(x, y, alpha):
    """min-sum 近似 f 运算"""
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    """

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def _hard_decision(self, L, R):
        """最左列硬判决"""
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        N, n, alpha = self.N, self.n, self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # 从右到左更新 L
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        r_idx = idx + s
                        L[idx, j - 1] = _f_min_sum(
                            R[idx, j - 1] + L[r_idx, j], L[idx, j], alpha
                        )
                        L[r_idx, j - 1] = (
                            _f_min_sum(R[idx, j - 1], L[idx, j], alpha) + L[r_idx, j]
                        )

            # 从左到右更新 R
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        r_idx = idx + s
                        R[idx, j + 1] = _f_min_sum(
                            R[r_idx, j] + L[r_idx, j + 1], R[idx, j], alpha
                        )
                        R[r_idx, j + 1] = (
                            _f_min_sum(R[idx, j], L[r_idx, j + 1], alpha) + R[r_idx, j]
                        )

            num_iters = it
            u_hat = self._hard_decision(L, R)

            # 早停：重编码与信道硬判决一致
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters
