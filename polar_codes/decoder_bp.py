"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


def _minsum_f(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        alpha = self.alpha
        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    Li = L[i, j]
                    Li_s = L[i + s, j]
                    Ri = R[i, j]
                    Li1 = L[i, j - 1]
                    Li1_s = L[i + s, j - 1]

                    L[i, j - 1] = _minsum_f(Ri + Li_s, Li1, alpha)
                    L[i + s, j - 1] = _minsum_f(Ri, Li1, alpha) + Li1_s

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    Ri = R[i, j]
                    Ri_s = R[i + s, j]
                    Li1 = L[i, j + 1]
                    Li1_s = L[i + s, j + 1]

                    R[i, j + 1] = _minsum_f(Ri_s + Li1_s, Ri, alpha)
                    R[i + s, j + 1] = _minsum_f(Ri, Li1, alpha) + Ri_s

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
