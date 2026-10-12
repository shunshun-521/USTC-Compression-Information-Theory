"""
极化码 BP（置信传播）译码器
因子图 min-sum，含早停
"""
import numpy as np
from encoder import polar_encode


def _minsum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_bits] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    Li = R[j, i] + L[j + 1, i + s]
                    Lj = L[j + 1, i]
                    L[j, i] = _minsum(Li, Lj, self.alpha)
                    L[j, i + s] = _minsum(R[j, i], Lj, self.alpha) + L[j + 1, i + s]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    Ri = R[j + 1, i + s] + L[j + 1, i + s]
                    R[j + 1, i] = _minsum(Ri, R[j, i], self.alpha)
                    R[j + 1, i + s] = _minsum(R[j, i], L[j + 1, i], self.alpha) + R[j, i + s]

            for i in range(N):
                u_hat[i] = 0 if (L[0, i] + R[0, i]) >= 0 else 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                break

        for i in range(N):
            u_hat[i] = 0 if (L[0, i] + R[0, i]) >= 0 else 1
        u_hat[self.frozen_bits] = 0
        return u_hat.astype(int), num_iters
