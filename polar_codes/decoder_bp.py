"""
极化码 BP（置信传播）译码器
基于因子图，min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（因子图 min-sum）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits).astype(bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for beta in range(s):
                        idx = i + beta
                        La = R[idx, j - 1] + L[idx + s, j]
                        Lb = L[idx, j]
                        L[idx, j - 1] = self._f_ms(La, Lb)
                        L[idx + s, j - 1] = self._f_ms(
                            R[idx, j - 1], L[idx, j]
                        ) + L[idx + s, j]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for beta in range(s):
                        idx = i + beta
                        Ra = R[idx + s, j] + L[idx + s, j + 1]
                        Rb = R[idx, j - 1] if j > 0 else R[idx, 0]
                        R[idx, j] = self._f_ms(Ra, Rb)
                        R[idx + s, j] = self._f_ms(
                            Rb, L[idx, j + 1]
                        ) + R[idx + s, j]

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break
            num_iters = it

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            u_hat[i] = 0 if (self.frozen_bits[i] or total >= 0) else 1

        return u_hat, num_iters
