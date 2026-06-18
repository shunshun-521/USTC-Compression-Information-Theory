"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def _f_ms(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for j in range(n - 1, -1, -1):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    for t in range(step):
                        i0 = i + t
                        i1 = i + t + step
                        L[i0, j] = self._f_ms(
                            R[i0, j] + L[i1, j + 1], L[i0, j + 1]
                        )
                        L[i1, j] = self._f_ms(R[i0, j], L[i0, j + 1]) + L[i1, j + 1]

            for j in range(1, n + 1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for t in range(step):
                        i0 = i + t
                        i1 = i + t + step
                        R[i0, j - 1] = self._f_ms(
                            R[i1, j] + L[i1, j], R[i0, j - 1]
                        )
                        R[i1, j - 1] = self._f_ms(
                            R[i0, j - 1], L[i0, j]
                        ) + R[i1, j]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
