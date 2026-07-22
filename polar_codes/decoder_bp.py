"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation
from channel import hard_decision_llr


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _f_min_sum(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        n = self.n
        N = self.N
        br = bit_reversal_permutation(N)
        llr = llr_ch[br].astype(np.float64)

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        idx = i + t
                        L[idx, j - 1] = self._f_min_sum(
                            R[idx, j - 1] + L[idx + s, j], L[idx, j]
                        )
                        L[idx + s, j - 1] = self._f_min_sum(
                            R[idx, j - 1], L[idx, j]
                        ) + L[idx + s, j]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        idx = i + t
                        R[idx, j + 1] = self._f_min_sum(
                            R[idx + s, j] + L[idx + s, j + 1], R[idx, j]
                        )
                        R[idx + s, j + 1] = self._f_min_sum(
                            R[idx, j], L[idx, j + 1]
                        ) + R[idx + s, j]

            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(int)

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_decision_llr(llr_ch)):
                num_iters = it
                break
        else:
            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(int)

        return u_hat, num_iters
