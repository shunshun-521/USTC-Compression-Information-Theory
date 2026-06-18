"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（因子图 n+1 列，列 0 为信源端，列 n 为信道端）。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def _f_min_sum(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # 右到左更新 L（列 n-1 到 0）
            for j in range(n - 1, -1, -1):
                s = 2 ** j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        L[idx, j] = self._f_min_sum(
                            R[idx, j + 1] + L[idx + s, j + 1],
                            L[idx, j + 1],
                        )
                        L[idx + s, j] = (
                            self._f_min_sum(R[idx, j + 1], L[idx, j + 1])
                            + L[idx + s, j + 1]
                        )

            # 左到右更新 R（列 1 到 n-1）
            for j in range(1, n):
                s = 2 ** (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        R[idx, j] = self._f_min_sum(
                            R[idx + s, j] + L[idx + s, j + 1],
                            R[idx, j - 1],
                        )
                        R[idx + s, j] = (
                            self._f_min_sum(R[idx, j - 1], L[idx, j + 1])
                            + R[idx + s, j]
                        )

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
