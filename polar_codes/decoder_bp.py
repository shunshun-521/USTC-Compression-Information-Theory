"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode


def _f_minsum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        n = self.n
        N = self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for j in range(n - 1, -1, -1):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        top = i + k
                        bottom = i + k + step
                        L[top, j] = _f_minsum(
                            R[top, j] + L[bottom, j + 1], L[top, j + 1], self.alpha
                        )
                        L[bottom, j] = _f_minsum(
                            R[top, j], L[top, j + 1], self.alpha
                        ) + L[bottom, j + 1]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        top = i + k
                        bottom = i + k + step
                        R[top, j + 1] = _f_minsum(
                            R[bottom, j + 1] + L[bottom, j + 1],
                            R[top, j],
                            self.alpha,
                        )
                        R[bottom, j + 1] = _f_minsum(
                            R[top, j], L[top, j + 1], self.alpha
                        ) + R[bottom, j + 1]

            u_hat = self._hard_decision(L, R)
            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break

        return self._hard_decision(L, R), num_iters

    def _hard_decision(self, L, R):
        u_hat = np.zeros(self.N, dtype=int)
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat
