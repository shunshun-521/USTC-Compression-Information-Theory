"""
极化码 BP（置信传播）译码器
min-sum 近似 + 早停
"""
import numpy as np

from encoder import polar_encode


class BPDecoder:
    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_min_sum(self, a, b):
        return self.alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    L[i, j - 1] = self._f_min_sum(
                        R[i, j] + L[i + step, j], L[i, j]
                    )
                    L[i + step, j - 1] = self._f_min_sum(
                        R[i, j], L[i, j]
                    ) + L[i + step, j]

            for j in range(1, n + 1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    R[i, j] = self._f_min_sum(
                        R[i + step, j] + L[i + step, j], R[i, j - 1]
                    )
                    R[i + step, j] = self._f_min_sum(
                        R[i, j - 1], L[i, j]
                    ) + R[i + step, j]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break

        return u_hat, num_iters
