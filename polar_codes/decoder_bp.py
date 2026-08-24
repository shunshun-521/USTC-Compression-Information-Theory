"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from channel import hard_decision_llr
from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    """min-sum f 函数"""
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 (u_hat, num_iters)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        alpha = self.alpha

        # L[i][j]: left messages, R[i][j]: right messages
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self._large

        u_hat = np.zeros(N, dtype=int)
        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            # 右到左更新 L
            for j in range(n, 0, -1):
                s = 2 ** (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_u = i + k
                        idx_l = i + k + s
                        L[idx_u, j - 1] = _f_min_sum(
                            R[idx_u, j] + L[idx_l, j], L[idx_u, j], alpha
                        )
                        L[idx_l, j - 1] = _f_min_sum(
                            R[idx_u, j], L[idx_u, j], alpha
                        ) + L[idx_l, j]

            # 左到右更新 R
            for j in range(0, n):
                s = 2 ** (j)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_u = i + k
                        idx_l = i + k + s
                        R[idx_u, j + 1] = _f_min_sum(
                            R[idx_l, j] + L[idx_l, j + 1], R[idx_u, j], alpha
                        )
                        R[idx_l, j + 1] = _f_min_sum(
                            R[idx_u, j], L[idx_u, j + 1], alpha
                        ) + R[idx_l, j]

            # 硬判决与早停
            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if self.frozen_bits[i] else (0 if total >= 0 else 1)

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            u_hat[i] = 0 if self.frozen_bits[i] else (0 if total >= 0 else 1)

        return u_hat.astype(int), num_iters
