"""
极化码 BP（置信传播）译码器，min-sum 近似 + 早停
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
        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        def hard_x(llr):
            return (llr < 0).astype(int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, step * 2):
                    L[i, j - 1] = self._f_ms(R[i, j] + L[i + step, j], L[i, j])
                    L[i + step, j - 1] = self._f_ms(R[i, j], L[i, j]) + L[i + step, j]
            for j in range(1, n + 1):
                step = 1 << (j - 1)
                for i in range(0, N, step * 2):
                    R[i, j] = self._f_ms(R[i + step, j] + L[i + step, j], R[i, j - 1])
                    R[i + step, j] = self._f_ms(R[i, j - 1], L[i, j]) + R[i + step, j]

            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                s = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if s >= 0 else 1
            u_hat[self.frozen_bits] = 0
            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_x(llr_ch)):
                return u_hat, it
        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            s = L[i, 0] + R[i, 0]
            u_hat[i] = 0 if s >= 0 else 1
        u_hat[self.frozen_bits] = 0
        return u_hat, self.max_iter
