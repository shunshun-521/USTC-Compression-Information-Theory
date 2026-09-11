"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import bit_reversal_permutation, polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（分层 min-sum，含早停）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)

    def _f_min_sum(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        br = self.br

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch[br]
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = 1e6

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(i, i + step):
                        L[k, j - 1] = self._f_min_sum(
                            R[k, j - 1] + L[k + step, j], L[k, j]
                        )
                        L[k + step, j - 1] = self._f_min_sum(
                            R[k, j - 1], L[k, j]
                        ) + L[k + step, j]

            for j in range(1, n + 1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(i, i + step):
                        R[k, j] = self._f_min_sum(
                            R[k + step, j - 1] + L[k + step, j], R[k, j - 1]
                        )
                        R[k + step, j] = self._f_min_sum(
                            R[k, j - 1], L[k, j]
                        ) + R[k + step, j - 1]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
