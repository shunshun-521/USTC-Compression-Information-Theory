"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode
from channel import hard_decision_llr


def _ms_f(a, b, alpha):
    """min-sum f 函数"""
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        N, n = self.N, self.n
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        frozen = self.frozen_bits

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[frozen, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # 从右到左更新 L
            for j in range(n, 0, -1):
                span = 1 << (j - 1)
                for i in range(0, N, 2 * span):
                    for k in range(span):
                        idx = i + k
                        idx2 = idx + span
                        L[idx, j - 1] = _ms_f(
                            R[idx, j] + L[idx2, j], L[idx, j], self.alpha
                        )
                        L[idx2, j - 1] = _ms_f(
                            R[idx, j], L[idx, j], self.alpha
                        ) + L[idx2, j]

            # 从左到右更新 R
            for j in range(1, n + 1):
                span = 1 << (j - 1)
                for i in range(0, N, 2 * span):
                    for k in range(span):
                        idx = i + k
                        idx2 = idx + span
                        R[idx, j] = _ms_f(
                            R[idx2, j] + L[idx2, j], R[idx, j - 1], self.alpha
                        )
                        R[idx2, j] = _ms_f(
                            R[idx, j - 1], L[idx, j], self.alpha
                        ) + R[idx2, j]

            # 判决与早停
            total = L[:, 0] + R[:, 0]
            u_hat = np.where(total >= 0, 0, 1).astype(int)
            u_hat[frozen] = 0

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_decision_llr(llr_ch)):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.where(total >= 0, 0, 1).astype(int)
        u_hat[frozen] = 0
        return u_hat, num_iters
