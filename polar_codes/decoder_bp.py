"""
极化码 BP（置信传播）译码器，min-sum + 早停
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.max_iter = max_iter
        self.alpha = alpha
        fb = np.asarray(frozen_bits)
        self.frozen = set(np.where(fb.astype(int) == 1)[0] if fb.dtype != bool else np.where(fb)[0])
        self.LARGE = 1e6

    def _f(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        for i in self.frozen:
            R[i, 0] = self.LARGE

        def hard_x(llr):
            return (llr < 0).astype(int)

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    L[i, j - 1] = self._f(R[i, j] + L[i + s, j], L[i, j])
                    L[i + s, j - 1] = self._f(R[i, j], L[i, j]) + L[i + s, j]
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    R[i, j + 1] = self._f(R[i + s, j] + L[i + s, j + 1], R[i, j])
                    R[i + s, j + 1] = self._f(R[i, j], L[i, j + 1]) + R[i + s, j]

            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
            for i in self.frozen:
                u_hat[i] = 0
            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_x(llr_ch)):
                num_iters = it
                break

        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
        for i in self.frozen:
            u_hat[i] = 0
        return u_hat.astype(int), num_iters
