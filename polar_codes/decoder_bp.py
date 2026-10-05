"""
极化码 BP（置信传播）译码器
min-sum 近似 + 早停
"""
import numpy as np
from encoder import polar_encode


class BPDecoder:
    """基于因子图的 BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_idx = set(np.where(self.frozen_bits)[0])
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_ms(self, a, b):
        return self.alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        for idx in self.frozen_idx:
            R[idx, 0] = self.LARGE

        hard = lambda llr: (llr < 0).astype(int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    L[i, j - 1] = self._f_ms(R[i, j] + L[i + s, j], L[i, j])
                    L[i + s, j - 1] = self._f_ms(R[i, j], L[i, j]) + L[i + s, j]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    R[i, j + 1] = self._f_ms(
                        R[i + s, j] + L[i + s, j + 1], R[i, j]
                    )
                    R[i + s, j + 1] = self._f_ms(R[i, j], L[i, j + 1]) + R[i + s, j]

            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                if i in self.frozen_idx:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard(llr_ch)):
                return u_hat, it

        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            u_hat[i] = 0 if i in self.frozen_idx or (L[i, 0] + R[i, 0]) >= 0 else 1
        return u_hat, self.max_iter
