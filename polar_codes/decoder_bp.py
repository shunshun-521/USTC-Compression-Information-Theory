"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits).astype(bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    @staticmethod
    def _f_min_sum(a, b, alpha):
        return alpha * np.sign(a) * np.sign(b) * min(abs(a), abs(b))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = [[0.0] * (n + 1) for _ in range(N)]
        R = [[0.0] * (n + 1) for _ in range(N)]

        for i in range(N):
            L[i][n] = llr_ch[i]
            R[i][0] = self.large if self.frozen_bits[i] else 0.0

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    s = step
                    la = R[i][j - 1] + L[i + s][j]
                    lb = L[i][j]
                    L[i][j] = self._f_min_sum(la, lb, self.alpha)
                    L[i + s][j] = self._f_min_sum(R[i][j - 1], lb, self.alpha) + L[i + s][j]

            for j in range(1, n + 1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    s = step
                    ra = R[i + s][j] + L[i + s][j]
                    rb = R[i][j - 1]
                    R[i][j] = self._f_min_sum(ra, rb, self.alpha)
                    R[i + s][j] = self._f_min_sum(rb, L[i][j], self.alpha) + R[i + s][j]

            for i in range(N):
                total = L[i][0] + R[i][0]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        for i in range(N):
            total = L[i][0] + R[i][0]
            u_hat[i] = 0 if self.frozen_bits[i] or total >= 0 else 1

        return u_hat, num_iters
