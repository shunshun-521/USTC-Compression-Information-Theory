"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)"""
        from encoder import bit_reversal_permutation

        llr_nat = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        llr = llr_nat[br]

        n, N = self.n, self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        u_hat = np.zeros(N, dtype=np.int8)
        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        L[idx, j - 1] = self._f_min_sum(
                            R[idx, j] + L[idx + s, j], L[idx, j]
                        )
                        L[idx + s, j - 1] = (
                            self._f_min_sum(R[idx, j], L[idx, j]) + L[idx + s, j]
                        )

            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        R[idx, j] = self._f_min_sum(
                            R[idx + s, j] + L[idx + s, j], R[idx, j - 1]
                        )
                        R[idx + s, j] = (
                            self._f_min_sum(R[idx, j - 1], L[idx, j]) + R[idx + s, j]
                        )

            total = L[:, 0] + R[:, 0]
            u_hat = np.where(total >= 0, 0, 1).astype(np.int8)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            x_hard = (llr_nat < 0).astype(np.int8)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

        return u_hat, num_iters
