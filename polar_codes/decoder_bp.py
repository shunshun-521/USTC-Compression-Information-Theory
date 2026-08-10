"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器，基于极化码因子图"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)"""
        n = self.n
        N = self.N
        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_ch

        for i in range(N):
            if self.frozen_bits[i]:
                R[0, i] = self.LARGE
            else:
                R[0, i] = 0.0

        num_iters = 0
        for it in range(self.max_iter):
            num_iters = it + 1

            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for block in range(0, N, 2 * step):
                    for i in range(step):
                        idx_l = block + i
                        idx_r = block + step + i
                        L[j - 1, idx_l] = self._f_min_sum(
                            R[j, idx_l] + L[j, idx_r],
                            L[j, idx_l],
                        )
                        L[j - 1, idx_r] = self._f_min_sum(
                            R[j, idx_l],
                            L[j, idx_l],
                        ) + L[j, idx_r]

            for j in range(0, n):
                step = 1 << j
                for block in range(0, N, 2 * step):
                    for i in range(step):
                        idx_l = block + i
                        idx_r = block + step + i
                        R[j + 1, idx_l] = self._f_min_sum(
                            R[j + 1, idx_r] + L[j + 1, idx_r],
                            R[j, idx_l],
                        )
                        R[j + 1, idx_r] = self._f_min_sum(
                            R[j, idx_l],
                            L[j + 1, idx_l],
                        ) + R[j + 1, idx_r]

            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                total = L[0, i] + R[0, i]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                return u_hat, num_iters

        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            total = L[0, i] + R[0, i]
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1
        return u_hat, num_iters
