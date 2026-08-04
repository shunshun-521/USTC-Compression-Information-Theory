"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_min_sum(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        """主译码函数"""
        N = self.N
        n = self.n

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)

        L[n, :] = llr_ch.copy()
        R[0, :] = 0.0
        R[0, self.frozen_bits] = self.LARGE

        num_iters = self.max_iter

        for it in range(self.max_iter):
            # 从信道侧向信源侧更新 L
            for stage in range(n - 1, -1, -1):
                bs = 2 ** stage
                for i in range(0, N, 2 * bs):
                    for j in range(bs):
                        L[stage, i + j] = self._f_min_sum(
                            R[stage, i + j] + L[stage + 1, i + j + bs],
                            L[stage + 1, i + j],
                        )
                        L[stage, i + j + bs] = self._f_min_sum(
                            R[stage, i + j],
                            L[stage + 1, i + j],
                        ) + L[stage + 1, i + j + bs]

            # 从信源侧向信道侧更新 R
            for stage in range(n):
                bs = 2 ** stage
                for i in range(0, N, 2 * bs):
                    for j in range(bs):
                        R[stage + 1, i + j] = self._f_min_sum(
                            R[stage + 1, i + j + bs] + L[stage + 1, i + j + bs],
                            R[stage, i + j],
                        )
                        R[stage + 1, i + j + bs] = self._f_min_sum(
                            R[stage, i + j],
                            L[stage + 1, i + j],
                        ) + R[stage + 1, i + j + bs]

            # 早停
            total = L[0, :] + R[0, :]
            u_hat = np.zeros(N, dtype=np.int8)
            u_hat[total < 0] = 1
            u_hat[self.frozen_bits] = 0
            x_hat = polar_encode(u_hat)
            hard = np.zeros(N, dtype=np.int8)
            hard[llr_ch < 0] = 1
            if np.array_equal(x_hat, hard):
                num_iters = it + 1
                break

        total = L[0, :] + R[0, :]
        u_hat = np.zeros(N, dtype=np.int8)
        u_hat[total < 0] = 1
        u_hat[self.frozen_bits] = 0

        return u_hat, num_iters
