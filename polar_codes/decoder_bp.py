"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（因子图列 0..m）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.m = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _f_ms(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        m = self.m
        N = self.N

        L = np.zeros((N, m + 1), dtype=np.float64)
        R = np.zeros((N, m + 1), dtype=np.float64)
        L[:, m] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits == 1, 0] = self.large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for stage in range(m - 1, -1, -1):
                step = 1 << stage
                for block in range(0, N, 2 * step):
                    for t in range(step):
                        i = block + t
                        j = stage
                        L[i, j] = self._f_ms(
                            R[i, j + 1] + L[i + step, j + 1],
                            L[i, j + 1],
                        )
                        L[i + step, j] = self._f_ms(R[i, j + 1], L[i, j + 1]) + L[i + step, j + 1]

            for stage in range(0, m):
                step = 1 << stage
                for block in range(0, N, 2 * step):
                    for t in range(step):
                        i = block + t
                        j = stage + 1
                        R[i, j] = self._f_ms(
                            R[i + step, j] + L[i + step, j],
                            R[i, j - 1],
                        )
                        R[i + step, j] = self._f_ms(R[i, j - 1], L[i, j]) + R[i + step, j]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits == 1] = 0

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break
        else:
            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits == 1] = 0

        return u_hat, num_iters
