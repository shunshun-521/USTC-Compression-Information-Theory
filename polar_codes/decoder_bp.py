"""
极化码 BP（置信传播）译码器：min-sum 近似 + 早停
"""
import numpy as np
import math

from encoder import polar_encode


def _f_ms(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（因子图 n+1 列）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((n + 1, N))
        R = np.zeros((n + 1, N))
        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_bits] = self.LARGE

        for it in range(self.max_iter):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for t in range(step):
                        i0, i1 = i + t, i + t + step
                        L[j - 1, i0] = _f_ms(
                            R[j, i0] + L[j, i1], L[j, i0], self.alpha
                        )
                        L[j - 1, i1] = _f_ms(R[j, i0], L[j, i0], self.alpha) + L[j, i1]

            for j in range(1, n + 1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for t in range(step):
                        i0, i1 = i + t, i + t + step
                        R[j, i0] = _f_ms(
                            R[j - 1, i0], L[j, i1], self.alpha
                        ) + R[j, i1]
                        R[j, i1] = _f_ms(R[j - 1, i0], L[j, i0], self.alpha) + R[j - 1, i1]

            u_hat = np.zeros(N, dtype=int)
            total = L[0, :] + R[0, :]
            u_hat[total < 0] = 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                return u_hat, it + 1

        u_hat = np.zeros(N, dtype=int)
        total = L[0, :] + R[0, :]
        u_hat[total < 0] = 1
        u_hat[self.frozen_bits] = 0
        return u_hat, self.max_iter
