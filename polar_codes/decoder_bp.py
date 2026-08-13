"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation
from channel import hard_decision_llr


class BPDecoder:
    """BP 译码器（因子图 min-sum，含早停）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        for it in range(1, self.max_iter + 1):
            # 右到左更新 L（列 n-1 .. 0）
            for j in range(n - 1, -1, -1):
                step = 1 << j
                for i in range(0, N, step << 1):
                    for k in range(step):
                        L[i + k, j] = self._f_min_sum(
                            R[i + k, j + 1] + L[i + k + step, j + 1],
                            L[i + k, j + 1],
                        )
                        L[i + k + step, j] = (
                            self._f_min_sum(R[i + k, j + 1], L[i + k, j + 1])
                            + L[i + k + step, j + 1]
                        )

            # 左到右更新 R（列 0 .. n-1）
            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, step << 1):
                    for k in range(step):
                        R[i + k, j + 1] = self._f_min_sum(
                            R[i + k + step, j + 1] + L[i + k + step, j + 1],
                            R[i + k, j],
                        )
                        R[i + k + step, j + 1] = (
                            self._f_min_sum(R[i + k, j], L[i + k, j + 1])
                            + R[i + k + step, j]
                        )

            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_decision_llr(llr_ch)):
                return u_hat.astype(int), it

        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat.astype(int), self.max_iter
