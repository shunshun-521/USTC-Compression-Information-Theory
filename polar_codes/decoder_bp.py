"""
极化码 BP（置信传播）译码器
"""
import numpy as np
from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（min-sum + 早停）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen, 0] = self._large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, step * 2):
                    for t in range(step):
                        i0, i1 = i + t, i + t + step
                        L[i0, j - 1] = _f_min_sum(
                            R[i0, j] + L[i1, j], L[i0, j], self.alpha
                        )
                        L[i1, j - 1] = _f_min_sum(
                            R[i0, j], L[i0, j], self.alpha
                        ) + L[i1, j]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, step * 2):
                    for t in range(step):
                        i0, i1 = i + t, i + t + step
                        R[i0, j + 1] = _f_min_sum(
                            R[i1, j] + L[i1, j + 1], R[i0, j], self.alpha
                        )
                        R[i1, j + 1] = _f_min_sum(
                            R[i0, j], L[i0, j + 1], self.alpha
                        ) + R[i1, j]

            u_hat = self._hard_decision(L, R)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        u_hat = self._hard_decision(L, R)
        return u_hat.astype(int), num_iters

    def _hard_decision(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen] = 0
        return u_hat
