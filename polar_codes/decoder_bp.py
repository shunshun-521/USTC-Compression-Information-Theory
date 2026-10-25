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

    def _f_ms(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        L = np.zeros((self.N, n + 1), dtype=np.float64)
        R = np.zeros((self.N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        frozen_idx = np.where(self.frozen_bits)[0]
        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 2 ** (j - 1)
                for i in range(0, self.N, 2 * step):
                    for k in range(step):
                        i1 = i + k
                        i2 = i + k + step
                        L[i1, j - 1] = self._f_ms(
                            R[i1, j] + L[i2, j], L[i1, j]
                        )
                        L[i2, j - 1] = self._f_ms(R[i1, j], L[i1, j]) + L[i2, j]

            for j in range(0, n):
                step = 2 ** (j)
                for i in range(0, self.N, 2 * step):
                    for k in range(step):
                        i1 = i + k
                        i2 = i + k + step
                        R[i1, j + 1] = self._f_ms(
                            R[i2, j] + L[i2, j + 1], R[i1, j]
                        )
                        R[i2, j + 1] = self._f_ms(R[i1, j], L[i1, j + 1]) + R[i2, j]

            u_hat = np.zeros(self.N, dtype=int)
            total = L[:, 0] + R[:, 0]
            u_hat[total < 0] = 1
            u_hat[frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        u_hat = np.zeros(self.N, dtype=int)
        total = L[:, 0] + R[:, 0]
        u_hat[total < 0] = 1
        u_hat[frozen_idx] = 0
        return u_hat, num_iters
