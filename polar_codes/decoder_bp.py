"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode


class BPDecoder:
    """BP 译码器（min-sum + 早停）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits).astype(bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6

    def _f_min_sum(self, a, b):
        return self.alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self._large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            for lam in range(n, 0, -1):
                step = 2 ** lam
                half = 2 ** (lam - 1)
                for phi in range(0, N, step):
                    for w in range(half):
                        a = phi + w
                        b = phi + w + half
                        L[a, lam - 1] = self._f_min_sum(R[a, lam] + L[b, lam], L[a, lam])
                        L[b, lam - 1] = self._f_min_sum(R[a, lam], L[a, lam]) + L[b, lam]

            for lam in range(1, n + 1):
                step = 2 ** lam
                half = 2 ** (lam - 1)
                for phi in range(0, N, step):
                    for w in range(half):
                        a = phi + w
                        b = phi + w + half
                        R[a, lam] = self._f_min_sum(R[b, lam - 1] + L[b, lam], R[a, lam - 1])
                        R[b, lam] = self._f_min_sum(R[a, lam - 1], L[a, lam]) + R[b, lam - 1]

            for i in range(N):
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                num_iters = it
                break

        return u_hat.astype(int), num_iters
