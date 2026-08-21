"""
极化码 BP（置信传播）译码器
"""
import math
import numpy as np
from decoder_sc import f_operation, align_llr_for_decoder
from encoder import polar_encode


class BPDecoder:
    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = align_llr_for_decoder(llr_ch)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    s = step
                    L[i, j - 1] = self._f_ms(R[i, j - 1] + L[i + s, j], L[i, j])
                    L[i + s, j - 1] = self._f_ms(R[i, j - 1], L[i, j]) + L[i + s, j]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    s = step
                    R[i, j] = self._f_ms(R[i + s, j] + L[i + s, j + 1], R[i, j - 1] if j > 0 else 0.0)
                    R[i + s, j] = self._f_ms(R[i, j - 1] if j > 0 else 0.0, L[i, j + 1]) + R[i + s, j]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                num_iters = it
                return u_hat, num_iters
            num_iters = it

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
