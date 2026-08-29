"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（列 0=信源，列 n=信道）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e7

    def _f(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        for it in range(1, self.max_iter + 1):
            L = np.zeros((N, n + 1), dtype=np.float64)
            R = np.zeros((N, n + 1), dtype=np.float64)
            L[:, n] = llr_ch
            R[:, 0] = 0.0
            R[self.frozen_bits, 0] = self.LARGE

            for j in range(n - 1, -1, -1):
                s = 1 << j
                L_next = L[:, j + 1].copy()
                R_next = R[:, j + 1].copy()
                for i in range(0, N, 2 * s):
                    L_next[i] = self._f(R[i, j + 1] + L[i + s, j + 1], L[i, j + 1])
                    L_next[i + s] = self._f(R[i, j + 1], L[i, j + 1]) + L[i + s, j + 1]
                L[:, j] = L_next + R[:, j + 1]

            for j in range(n):
                s = 1 << j
                R_next = R[:, j].copy()
                for i in range(0, N, 2 * s):
                    R_next[i] = self._f(R[i + s, j + 1] + L[i + s, j + 1], R[i, j])
                    R_next[i + s] = self._f(R[i, j], L[i, j + 1]) + R[i + s, j]
                R[:, j + 1] = R_next + L[:, j + 1]

            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[total < 0] = 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                return u_hat, it

        return u_hat, self.max_iter
