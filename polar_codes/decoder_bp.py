"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation


class BPDecoder:
    """BP 译码器（极化码因子图，min-sum + 早停）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _f_minsum(self, x, y):
        sign = np.sign(x) * np.sign(y)
        mag = self.alpha * np.minimum(np.abs(x), np.abs(y))
        return sign * mag

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        frozen_idx = np.where(self.frozen_bits)[0]
        R[frozen_idx, 0] = self.large

        br = bit_reversal_permutation(N)
        llr_nat = llr_ch[br]
        num_iters = 0

        for it in range(self.max_iter):
            num_iters = it + 1

            for j in range(n - 1, -1, -1):
                span = 1 << j
                for i in range(0, N, 2 * span):
                    for k in range(span):
                        idx_u = i + k
                        idx_v = i + k + span
                        L[idx_u, j] = self._f_minsum(
                            R[idx_u, j] + L[idx_v, j + 1], L[idx_u, j + 1]
                        )
                        L[idx_v, j] = self._f_minsum(
                            R[idx_u, j], L[idx_u, j + 1]
                        ) + L[idx_v, j + 1]

            for j in range(1, n + 1):
                span = 1 << (j - 1)
                for i in range(0, N, 2 * span):
                    for k in range(span):
                        idx_u = i + k
                        idx_v = i + k + span
                        R[idx_v, j] = self._f_minsum(
                            R[idx_v, j - 1] + L[idx_v, j], R[idx_u, j - 1]
                        )
                        R[idx_u, j] = self._f_minsum(
                            R[idx_u, j - 1], L[idx_v, j]
                        ) + R[idx_v, j - 1]

            total = L[:, 0] + R[:, 0]
            u_hat = np.where(total >= 0, 0, 1).astype(int)
            u_hat[frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            x_hard = (llr_nat < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.where(total >= 0, 0, 1).astype(int)
        u_hat[frozen_idx] = 0
        return u_hat, num_iters
