"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器（因子图 min-sum，含早停）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e8

    def _f_min_sum(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        # 列 0 为信道端，列 n 为信源端（与部分文献一致的索引）
        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[0, :] = llr_ch
        R[n, :] = 0.0
        R[n, self.frozen_bits] = self._large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for stage in range(1, n + 1):
                step = 1 << (stage - 1)
                for base in range(0, N, step << 1):
                    for k in range(step):
                        i = base + k
                        ip = i + step
                        L[stage, i] = self._f_min_sum(
                            L[stage - 1, i],
                            L[stage - 1, ip] + R[stage, ip],
                        )
                        L[stage, ip] = self._f_min_sum(
                            R[stage, i],
                            L[stage - 1, i],
                        ) + L[stage - 1, ip]

            for stage in range(n - 1, -1, -1):
                step = 1 << stage
                for base in range(0, N, step << 1):
                    for k in range(step):
                        i = base + k
                        ip = i + step
                        R[stage, i] = self._f_min_sum(
                            R[stage + 1, i],
                            L[stage, ip] + R[stage + 1, ip],
                        )
                        R[stage, ip] = self._f_min_sum(
                            R[stage + 1, i],
                            L[stage, i],
                        ) + R[stage + 1, ip]

            total = L[n, :] + R[n, :]
            u_hat = np.where(total >= 0, 0, 1).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[n, :] + R[n, :]
        u_hat = np.where(total >= 0, 0, 1).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
