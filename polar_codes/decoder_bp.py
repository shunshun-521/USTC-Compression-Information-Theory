"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    """min-sum 近似 f 运算。"""
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """
    BP 译码器。
    L[stage, i]、R[stage, i]：stage=0 为信源端，stage=n 为信道端。
    """

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 u_hat, num_iters
        """
        N = self.N
        m = self.n
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        L = np.zeros((m + 1, N), dtype=np.float64)
        R = np.zeros((m + 1, N), dtype=np.float64)

        L[m, :] = llr_ch
        R[0, self.frozen_idx] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # 左向更新 L：stage m-1 -> 0
            for stage in range(m - 1, -1, -1):
                bs = 1 << (stage + 1)
                half = bs // 2
                for base in range(0, N, bs):
                    for k in range(half):
                        i = base + k
                        j = base + k + half
                        L[stage, i] = _f_min_sum(
                            L[stage + 1, i],
                            L[stage + 1, j] + R[stage, j],
                            self.alpha,
                        )
                        L[stage, j] = (
                            _f_min_sum(R[stage, i], L[stage + 1, i], self.alpha)
                            + L[stage + 1, j]
                        )

            # 右向更新 R：stage 0 -> m-1
            for stage in range(m):
                bs = 1 << (stage + 1)
                half = bs // 2
                for base in range(0, N, bs):
                    for k in range(half):
                        i = base + k
                        j = base + k + half
                        R[stage + 1, i] = _f_min_sum(
                            R[stage, i],
                            L[stage + 1, i] + R[stage, j],
                            self.alpha,
                        )
                        R[stage + 1, j] = (
                            _f_min_sum(R[stage, i], L[stage + 1, i], self.alpha)
                            + R[stage, j]
                        )

            total = L[0, :] + R[0, :]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[0, :] + R[0, :]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
