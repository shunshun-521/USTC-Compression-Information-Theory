"""
极化码 BP（置信传播）译码器
"""
import numpy as np
import math

from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        for it in range(1, self.max_iter + 1):
            for ell in range(n - 1, -1, -1):
                s = 2 ** ell
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        L[i + t, ell] = self._f_ms(
                            R[i + t, ell + 1] + L[i + t + s, ell + 1],
                            L[i + t, ell + 1],
                        )
                        L[i + t + s, ell] = (
                            self._f_ms(R[i + t, ell + 1], L[i + t, ell + 1])
                            + L[i + t + s, ell + 1]
                        )

            for ell in range(0, n):
                s = 2 ** ell
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        R[i + t, ell + 1] = self._f_ms(
                            R[i + t + s, ell + 1] + L[i + t + s, ell + 1],
                            R[i + t, ell],
                        )
                        R[i + t + s, ell + 1] = (
                            self._f_ms(R[i + t, ell], L[i + t, ell + 1])
                            + R[i + t + s, ell]
                        )

            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[total < 0] = 1
            u_hat[self.frozen_bits] = 0
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                return u_hat, it

        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[total < 0] = 1
        u_hat[self.frozen_bits] = 0
        return u_hat, self.max_iter
