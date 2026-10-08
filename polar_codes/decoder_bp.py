"""
极化码 BP（置信传播）译码器，min-sum 近似 + 早停
"""
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_ms(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        from encoder import bit_reversal_permutation

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        llr = llr_ch[br]

        n = self.n
        N = self.N
        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))
        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 2 ** (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(i, i + s):
                        L[k, j - 1] = self._f_ms(
                            R[k, j] + L[k + s, j], L[k, j + 1]
                        )
                        L[k + s, j - 1] = self._f_ms(R[k, j], L[k, j + 1]) + L[k + s, j + 1]

            for j in range(0, n):
                s = 2 ** (j)
                for i in range(0, N, 2 * s):
                    for k in range(i, i + s):
                        R[k, j + 1] = self._f_ms(
                            R[k + s, j] + L[k + s, j + 1], R[k, j]
                        )
                        R[k + s, j + 1] = self._f_ms(R[k, j], L[k, j + 1]) + R[k + s, j]

            for i in range(N):
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                num_iters = it
                break
            num_iters = it

        for i in range(N):
            u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
        u_hat[self.frozen_idx] = 0
        inv = np.zeros(N, dtype=int)
        inv[br] = np.arange(N)
        u_nat = u_hat[inv]
        return u_nat.astype(int), num_iters
