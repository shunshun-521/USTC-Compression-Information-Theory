"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode


def _ms_f(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（min-sum + 早停）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        hard_ch = (llr_ch < 0).astype(int)
        num_iters = 0

        for it in range(1, self.max_iter + 1):
            num_iters = it
            for j in range(n, 0, -1):
                step = 2 ** (j - 1)
                for i in range(0, N, 2 * step):
                    for t in range(step):
                        a = i + t
                        b = i + t + step
                        L[a, j - 1] = _ms_f(
                            R[a, j] + L[b, j], L[a, j], self.alpha
                        )
                        L[b, j - 1] = _ms_f(R[a, j], L[a, j], self.alpha) + L[b, j]

            for j in range(0, n):
                step = 2 ** j
                for i in range(0, N, 2 * step):
                    for t in range(step):
                        a = i + t
                        b = i + t + step
                        R[a, j + 1] = _ms_f(
                            R[b, j] + L[b, j + 1], R[a, j], self.alpha
                        )
                        R[b, j + 1] = _ms_f(R[a, j], L[a, j + 1], self.alpha) + R[b, j]

            u_hat = (L[:, 0] + R[:, 0] < 0).astype(int)
            u_hat[self.frozen_bits] = 0
            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_ch):
                break

        u_hat = (L[:, 0] + R[:, 0] < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
