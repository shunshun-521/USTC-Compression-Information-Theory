"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation, _permute_channel_llr


class BPDecoder:
    """BP 译码器（因子图 min-sum）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6

    def _f_min_sum(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = _permute_channel_llr(llr_ch)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self._large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    L[i, j - 1] = self._f_min_sum(
                        R[i, j] + L[i + s, j], L[i, j]
                    )
                    L[i + s, j - 1] = self._f_min_sum(
                        R[i, j], L[i, j]
                    ) + L[i + s, j]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    R[i, j + 1] = self._f_min_sum(
                        R[i + s, j] + L[i + s, j + 1], R[i, j]
                    )
                    R[i + s, j + 1] = self._f_min_sum(
                        R[i, j], L[i, j + 1]
                    ) + R[i + s, j]

            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break

        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            total = L[i, 0] + R[i, 0]
            u_hat[i] = 0 if total >= 0 else 1
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
