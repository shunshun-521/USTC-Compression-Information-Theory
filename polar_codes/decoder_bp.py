"""
极化码 BP（置信传播）译码器，min-sum，含早停
"""
import numpy as np
from encoder import polar_encode


def _f_minsum(a, b, alpha):
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

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        hard = lambda llr: (llr < 0).astype(int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for t in range(step):
                        idx = i + t
                        L[idx, j - 1] = _f_minsum(
                            R[idx, j] + L[idx + step, j], L[idx, j], self.alpha
                        )
                        L[idx + step, j - 1] = _f_minsum(
                            R[idx, j], L[idx, j], self.alpha
                        ) + L[idx + step, j]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    for t in range(step):
                        idx = i + t
                        R[idx, j + 1] = _f_minsum(
                            R[idx + step, j] + L[idx + step, j + 1], R[idx, j], self.alpha
                        )
                        R[idx + step, j + 1] = _f_minsum(
                            R[idx, j], L[idx, j + 1], self.alpha
                        ) + R[idx + step, j]

            u_hat = hard(L[:, 0] + R[:, 0])
            u_hat[self.frozen_bits] = 0
            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard(llr_ch)):
                return u_hat.astype(int), it

        u_hat = hard(L[:, 0] + R[:, 0])
        u_hat[self.frozen_bits] = 0
        return u_hat.astype(int), self.max_iter
