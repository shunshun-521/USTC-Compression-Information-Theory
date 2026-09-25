"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.L = np.zeros((N, self.n + 1), dtype=np.float64)
        self.R = np.zeros((N, self.n + 1), dtype=np.float64)

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.clip(np.asarray(llr_ch, dtype=np.float64), -1e2, 1e2)
        n = self.n
        N = self.N

        self.L.fill(0.0)
        self.R.fill(0.0)
        self.L[:, n] = llr_ch
        self.R[:, 0] = 0.0
        self.R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    self.L[i, j - 1] = self._f_min_sum(
                        self.R[i, j] + self.L[i + step, j],
                        self.L[i, j],
                    )
                    self.L[i + step, j - 1] = self._f_min_sum(
                        self.R[i, j], self.L[i, j]
                    ) + self.L[i + step, j]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    self.R[i, j + 1] = self._f_min_sum(
                        self.R[i + step, j] + self.L[i + step, j + 1],
                        self.R[i, j],
                    )
                    self.R[i + step, j + 1] = self._f_min_sum(
                        self.R[i, j], self.L[i + step, j + 1]
                    ) + self.R[i + step, j]

            total = self.L[:, 0] + self.R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0
            num_iters = it

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                break

        total = self.L[:, 0] + self.R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
