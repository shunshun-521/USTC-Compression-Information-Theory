"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode


def _f_min_sum(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器"""

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

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for lev in range(n - 1, -1, -1):
                s = 1 << lev
                for i in range(0, N, 2 * s):
                    for k in range(i, i + s):
                        L[k, lev] = _f_min_sum(
                            R[k, lev] + L[k + s, lev + 1], L[k, lev + 1], self.alpha
                        )
                        L[k + s, lev] = (
                            _f_min_sum(R[k, lev], L[k, lev + 1], self.alpha) + L[k + s, lev + 1]
                        )

            for lev in range(n):
                s = 1 << lev
                for i in range(0, N, 2 * s):
                    for k in range(i, i + s):
                        R[k, lev + 1] = _f_min_sum(
                            R[k + s, lev] + L[k + s, lev + 1], R[k, lev], self.alpha
                        )
                        R[k + s, lev + 1] = (
                            _f_min_sum(R[k, lev], L[k, lev + 1], self.alpha) + R[k + s, lev]
                        )

            for i in range(N):
                u_hat[i] = 0 if self.frozen_bits[i] else (0 if (L[i, 0] + R[i, 0]) >= 0 else 1)

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            u_hat[i] = 0 if self.frozen_bits[i] else (0 if (L[i, 0] + R[i, 0]) >= 0 else 1)

        return u_hat, num_iters
