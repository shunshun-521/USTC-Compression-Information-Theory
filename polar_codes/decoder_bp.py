"""
极化码 BP（置信传播）译码器
"""
import numpy as np
from encoder import polar_encode


class BPDecoder:
    """BP 译码器（min-sum，早停）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.max_iter = max_iter
        self.alpha = alpha
        frozen_bits = np.asarray(frozen_bits)
        self.frozen_bits = frozen_bits.astype(bool)
        self.large = 1e6

    def _f_ms(self, x, y):
        sx = np.where(x >= 0, 1.0, -1.0)
        sy = np.where(y >= 0, 1.0, -1.0)
        return self.alpha * sx * sy * np.minimum(np.abs(x), np.abs(y))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 2 ** (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        L[idx, j - 1] = self._f_ms(R[idx, j] + L[idx + s, j], L[idx, j])
                        L[idx + s, j - 1] = self._f_ms(R[idx, j], L[idx, j]) + L[idx + s, j]

            for j in range(0, n):
                s = 2 ** (j)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        R[idx, j + 1] = self._f_ms(R[idx + s, j] + L[idx + s, j + 1], R[idx, j])
                        R[idx + s, j + 1] = self._f_ms(R[idx, j], L[idx, j + 1]) + R[idx + s, j]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
