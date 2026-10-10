"""
极化码 BP（置信传播）译码器，min-sum 近似，含早停
"""
import numpy as np

from encoder import polar_encode


def _f_min_sum(x, y, alpha):
    sx = np.sign(x)
    sy = np.sign(y)
    sx = np.where(sx == 0, 1.0, sx)
    sy = np.where(sy == 0, 1.0, sy)
    return alpha * sx * sy * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        if self.frozen_bits.dtype != bool:
            self.frozen_bits = self.frozen_bits.astype(np.int32) == 1
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        for it in range(self.max_iter):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for b in range(s):
                        i0 = i + b
                        i1 = i + b + s
                        L[i0, j - 1] = _f_min_sum(
                            R[i0, j] + L[i1, j], L[i0, j], self.alpha
                        )
                        L[i1, j - 1] = _f_min_sum(
                            R[i0, j], L[i0, j], self.alpha
                        ) + L[i1, j]

            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for b in range(s):
                        i0 = i + b
                        i1 = i + b + s
                        R[i0, j - 1] = _f_min_sum(
                            R[i1, j] + L[i1, j], R[i0, j - 1], self.alpha
                        )
                        R[i1, j - 1] = _f_min_sum(
                            R[i0, j - 1], L[i0, j], self.alpha
                        ) + R[i1, j]

            u_hat = np.zeros(N, dtype=int)
            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                return u_hat, it + 1

        u_hat = (L[:, 0] + R[:, 0] < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, self.max_iter
