"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import f_operation_min_sum
from encoder import polar_encode


class BPDecoder:
    """BP 译码器（因子图 flooding schedule）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.LARGE = 1e6

    def _f(self, a, b):
        return self.alpha * f_operation_min_sum(a, b)

    def decode(self, llr_ch):
        """主译码函数。"""
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch.copy()
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            L[:, n] = llr_ch.copy()

            R_new = R.copy()
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    R_new[i, j + 1] = self._f(
                        R[i + s, j] + L[i + s, j + 1], R[i, j]
                    )
                    R_new[i + s, j + 1] = self._f(
                        R[i, j], L[i, j + 1]
                    ) + R[i + s, j]
            R = R_new

            L_new = L.copy()
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    L_new[i, j - 1] = self._f(
                        R[i, j] + L[i + s, j], L[i, j]
                    )
                    L_new[i + s, j - 1] = self._f(
                        R[i, j], L[i, j]
                    ) + L[i + s, j]
            L = L_new

            for i in range(N):
                u_hat[i] = 0 if (self.frozen_bits[i] or (L[i, 0] + R[i, 0]) >= 0) else 1

            if np.array_equal(polar_encode(u_hat), (llr_ch < 0).astype(int)):
                num_iters = it
                break

        for i in range(N):
            u_hat[i] = 0 if (self.frozen_bits[i] or (L[i, 0] + R[i, 0]) >= 0) else 1

        return u_hat, num_iters
