"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from channel import hard_decision_llr
from encoder import polar_encode


def _boxplus_min_sum(a, b, alpha=1.0):
    """min-sum box-plus 运算。"""
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），列 n 为信道 LLR。
    """

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 (u_hat, num_iters)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=np.int32)

        for it in range(1, self.max_iter + 1):
            # 从右到左更新 L 消息（列 n -> 1）
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        L[idx, j - 1] = _boxplus_min_sum(
                            R[idx, j] + L[idx + s, j], L[idx, j], alpha
                        )
                        L[idx + s, j - 1] = _boxplus_min_sum(
                            R[idx, j], L[idx, j], alpha
                        ) + L[idx + s, j]

            # 从左到右更新 R 消息（列 0 -> n-1）
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        R[idx, j + 1] = _boxplus_min_sum(
                            R[idx + s, j + 1] + L[idx + s, j + 1], R[idx, j], alpha
                        )
                        R[idx + s, j + 1] = _boxplus_min_sum(
                            R[idx, j], L[idx, j + 1], alpha
                        ) + R[idx + s, j + 1]

            u_hat = np.zeros(N, dtype=np.int32)
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

        return u_hat, num_iters
