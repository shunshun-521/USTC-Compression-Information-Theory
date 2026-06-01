"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器（因子图 n+1 列，列 n 为信道 LLR）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int).astype(bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self._large = 1e6

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
        R[self.frozen_idx, 0] = self._large

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it
            for stage in range(n):
                s = 1 << stage
                jp = stage + 1
                j = stage
                for i in range(0, N, 2 * s):
                    L[i, j] = self._f_min_sum(
                        R[i, jp] + L[i + s, jp], L[i, jp]
                    )
                    L[i + s, j] = (
                        self._f_min_sum(R[i, jp], L[i, jp]) + L[i + s, jp]
                    )

            for stage in range(n):
                s = 1 << stage
                j = stage
                jp = stage + 1
                for i in range(0, N, 2 * s):
                    R[i, jp] = self._f_min_sum(
                        R[i + s, j] + L[i + s, jp], R[i, j]
                    )
                    R[i + s, jp] = (
                        self._f_min_sum(R[i, j], L[i, jp]) + R[i + s, j]
                    )

            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                break

        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
