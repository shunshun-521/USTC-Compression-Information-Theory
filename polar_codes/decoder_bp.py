"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math
from decoder_sc import f_operation_minsum
from encoder import polar_encode


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation_minsum(a, b)

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, num_iters"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_i = i + k
                        idx_is = i + k + s
                        L[idx_i, j - 1] = self._f_min_sum(
                            R[idx_i, j] + L[idx_is, j],
                            L[idx_i, j]
                        )
                        L[idx_is, j - 1] = self._f_min_sum(
                            R[idx_i, j],
                            L[idx_i, j]
                        ) + L[idx_is, j]

            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_i = i + k
                        idx_is = i + k + s
                        R[idx_i, j] = self._f_min_sum(
                            R[idx_is, j] + L[idx_is, j],
                            R[idx_i, j - 1]
                        )
                        R[idx_is, j] = self._f_min_sum(
                            R[idx_i, j - 1],
                            L[idx_i, j]
                        ) + R[idx_is, j]

            num_iters = it

            for i in range(N):
                u_hat[i] = 0 if self.frozen_bits[i] else (0 if (L[i, 0] + R[i, 0]) >= 0 else 1)

            x_hat = polar_encode(u_hat)
            hard_decision = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_decision):
                break

        for i in range(N):
            u_hat[i] = 0 if self.frozen_bits[i] else (0 if (L[i, 0] + R[i, 0]) >= 0 else 1)

        return u_hat, num_iters
