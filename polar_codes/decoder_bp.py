"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode

LARGE = 1e6


def _f_minsum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((n + 1, N))
        R = np.zeros((n + 1, N))
        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_bits] = LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for s in range(n, 0, -1):
                step = 2 ** (s - 1)
                for phi in range(0, N, 2 * step):
                    for beta in range(step):
                        i = phi + beta
                        j = i + step
                        L[s - 1, i] = _f_minsum(
                            R[i, s] + L[j, s], L[i, s], self.alpha
                        )
                        L[s - 1, j] = _f_minsum(R[i, s], L[i, s], self.alpha) + L[j, s]

            for s in range(1, n + 1):
                step = 2 ** (s - 1)
                for phi in range(0, N, 2 * step):
                    for beta in range(step):
                        i = phi + beta
                        j = i + step
                        R[j, s] = _f_minsum(
                            R[j, s - 1] + L[j, s], R[i, s - 1], self.alpha
                        ) + L[j, s]
                        R[i, s] = _f_minsum(R[i, s - 1], L[i, s], self.alpha) + R[j, s - 1]

            for i in range(N):
                total = L[0, i] + R[0, i]
                u_hat[i] = 0 if total >= 0 else 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break

        return u_hat, num_iters
