"""
极化码 BP（置信传播）译码器，min-sum 近似 + 早停
"""
import numpy as np
from encoder import polar_encode
from channel import codeword_channel_to_decoder


class BPDecoder:
    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.max_iter = max_iter
        self.alpha = alpha
        fb = np.asarray(frozen_bits)
        self.frozen = fb.astype(bool) if fb.dtype == bool else (fb.astype(int) != 0)
        self.frozen_idx = np.where(self.frozen)[0]
        self.LARGE = 1e6

    def _f_minsum(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        sign = np.sign(x) * np.sign(y)
        sign = np.where(sign == 0, 1.0, sign)
        return self.alpha * sign * np.minimum(np.abs(x), np.abs(y))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N
        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    L[i, j] = self._f_minsum(R[i, j + 1] + L[i + s, j + 1], L[i, j + 1])
                    L[i + s, j] = self._f_minsum(R[i, j + 1], L[i, j + 1]) + L[i + s, j + 1]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    R[i, j + 1] = self._f_minsum(R[i + s, j] + L[i + s, j + 1], R[i, j])
                    R[i + s, j + 1] = self._f_minsum(R[i, j], L[i, j + 1]) + R[i + s, j]

            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
            u_hat[self.frozen_idx] = 0
            x_hat = codeword_channel_to_decoder(polar_encode(u_hat))
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
