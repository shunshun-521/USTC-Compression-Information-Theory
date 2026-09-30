"""
极化码 BP（置信传播）译码器
min-sum 近似 + 早停
"""
import numpy as np
from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_mask = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N
        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_mask, 0] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, step << 1):
                    for t in range(step):
                        i1 = i + t
                        i2 = i + t + step
                        L[i1, j - 1] = _f_min_sum(
                            R[i1, j] + L[i2, j], L[i1, j], self.alpha
                        )
                        L[i2, j - 1] = _f_min_sum(R[i1, j], L[i1, j], self.alpha) + L[i2, j]

            for j in range(1, n + 1):
                step = 1 << (j - 1)
                for i in range(0, N, step << 1):
                    for t in range(step):
                        i1 = i + t
                        i2 = i + t + step
                        R[i1, j] = _f_min_sum(
                            R[i2, j] + L[i2, j], R[i1, j - 1], self.alpha
                        )
                        R[i2, j] = _f_min_sum(R[i1, j - 1], L[i1, j], self.alpha) + R[i2, j]

            u_hat = (L[:, 0] + R[:, 0] >= 0).astype(int)
            u_hat[self.frozen_mask] = 0
            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            num_iters = it
            if np.array_equal(x_hat, hard_x):
                break

        u_hat[self.frozen_mask] = 0
        return u_hat, num_iters
