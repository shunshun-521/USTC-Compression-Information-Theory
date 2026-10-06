"""
极化码 BP（置信传播）译码器
基于因子图，min-sum 近似，含早停机制
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
        N = self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for t in range(step):
                        a = i + t
                        b = a + step
                        R[a, j] = self._f_ms(R[b, j] + L[b, j + 1], L[a, j + 1])
                        L[a, j] = self._f_ms(R[a, j], L[a, j + 1]) + L[b, j + 1]
                        L[b, j] = self._f_ms(R[a, j], L[a, j + 1]) + R[b, j]

            for j in range(1, n + 1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for t in range(step):
                        a = i + t
                        b = a + step
                        R[a, j - 1] = self._f_ms(R[b, j] + L[b, j], R[a, j - 1])
                        R[b, j - 1] = self._f_ms(R[a, j - 1], L[a, j]) + R[b, j]

            u_hat = np.zeros(N, dtype=np.int8)
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break

        u_hat = np.zeros(N, dtype=np.int8)
        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
        return u_hat, num_iters
