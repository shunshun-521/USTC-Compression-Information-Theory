"""
极化码 BP（置信传播）译码器：min-sum + 早停
"""
import numpy as np
import math

from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_ms(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen, 0] = self.LARGE

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, step << 1):
                    for b in range(step):
                        idx = i + b
                        L[idx, j - 1] = self._f_ms(
                            R[idx, j] + L[idx + step, j], L[idx, j]
                        )
                        L[idx + step, j - 1] = self._f_ms(R[idx, j], L[idx, j]) + L[idx + step, j]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, step << 1):
                    for b in range(step):
                        idx = i + b
                        R[idx, j + 1] = self._f_ms(
                            R[idx + step, j] + L[idx + step, j + 1], R[idx, j]
                        )
                        R[idx + step, j + 1] = self._f_ms(R[idx, j], L[idx, j + 1]) + R[idx + step, j]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen] = 0
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen] = 0
        return u_hat, num_iters
