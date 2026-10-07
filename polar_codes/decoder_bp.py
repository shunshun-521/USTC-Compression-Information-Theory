"""
极化码 BP（置信传播）译码器：min-sum + 早停
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N
        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_bits] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 2 ** (j - 1)
                for i in range(0, N, 2 * step):
                    for t in range(step):
                        a = R[i + t, j - 1] + L[i + t + step, j]
                        b = L[i + t, j]
                        L[i + t, j - 1] = self._f_ms(a, b)
                        c = R[i + t, j - 1]
                        d = L[i + t, j]
                        e = L[i + t + step, j]
                        L[i + t + step, j - 1] = self._f_ms(c, d) + e

            for j in range(0, n):
                step = 2 ** (j)
                for i in range(0, N, 2 * step):
                    for t in range(step):
                        a = R[i + t + step, j] + L[i + t + step, j + 1]
                        b = R[i + t, j]
                        R[i + t, j + 1] = self._f_ms(a, b)
                        c = R[i + t, j]
                        d = L[i + t, j + 1]
                        e = R[i + t + step, j]
                        R[i + t + step, j + 1] = self._f_ms(c, d) + e

            num_iters = it
            total = L[0, :] + R[0, :]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0
            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                break

        total = L[0, :] + R[0, :]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
