"""
极化码 BP（置信传播）译码器
基于因子图 min-sum，含早停机制
"""
import numpy as np
import math
from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.LARGE = 1e7

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            R[:, 0] = 0.0
            R[self.frozen_idx, 0] = self.LARGE

            L_old = L.copy()
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        a = i + k
                        b = a + s
                        L[a, j] = _f_min_sum(
                            R[a, j] + L_old[b, j + 1], L_old[a, j + 1], self.alpha
                        )
                        L[b, j] = (
                            _f_min_sum(R[a, j], L_old[a, j + 1], self.alpha)
                            + L_old[b, j + 1]
                        )

            R_old = R.copy()
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        a = i + k
                        b = a + s
                        R[a, j + 1] = _f_min_sum(
                            R_old[b, j] + L[b, j + 1], R_old[a, j], self.alpha
                        )
                        R[b, j + 1] = (
                            _f_min_sum(R_old[a, j], L[a, j + 1], self.alpha) + R_old[b, j]
                        )

            num_iters = it
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        return u_hat, num_iters
