"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode


def _f_min_sum(x, y, alpha):
    """min-sum f 运算，带 alpha 修正。"""
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """
    BP 译码器。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.LARGE = 1e6

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        alpha = self.alpha

        # L[i][j]: shape (N, n+1), L messages from right to left
        # R[i][j]: shape (N, n+1), R messages from left to right
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter

        for it in range(self.max_iter):
            # 从右到左更新 L 消息
            for j in range(n, 0, -1):
                s = 2 ** (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        idx_s = i + k + s
                        L[idx, j - 1] = _f_min_sum(
                            R[idx, j] + L[idx_s, j], L[idx, j], alpha
                        )
                        L[idx_s, j - 1] = _f_min_sum(R[idx, j], L[idx, j], alpha) + L[idx_s, j]

            # 从左到右更新 R 消息
            for j in range(1, n + 1):
                s = 2 ** (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        idx_s = i + k + s
                        R[idx, j] = _f_min_sum(
                            R[idx_s, j] + L[idx_s, j], R[idx, j - 1], alpha
                        )
                        R[idx_s, j] = _f_min_sum(R[idx, j - 1], L[idx, j], alpha) + R[idx_s, j]

            # 早停检查
            total_llr = L[:, 0] + R[:, 0]
            u_hat = np.where(total_llr >= 0, 0, 1).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                num_iters = it + 1
                break

        total_llr = L[:, 0] + R[:, 0]
        u_hat = np.where(total_llr >= 0, 0, 1).astype(int)
        u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
