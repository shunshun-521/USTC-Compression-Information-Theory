"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（因子图 min-sum）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.rev = bit_reversal_permutation(N)
        self.large = 1e6

    def _f_min_sum(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        N, n = self.N, self.n
        llr = np.asarray(llr_ch, dtype=np.float64)[self.rev]

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, step << 1):
                    for k in range(step):
                        idx = i + k
                        partner = idx + step
                        L[idx, j - 1] = self._f_min_sum(
                            R[idx, j] + L[partner, j], L[idx, j]
                        )
                        L[partner, j - 1] = self._f_min_sum(
                            R[idx, j], L[idx, j]
                        ) + L[partner, j]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, step << 1):
                    for k in range(step):
                        idx = i + k
                        partner = idx + step
                        R[idx, j + 1] = self._f_min_sum(
                            R[partner, j] + L[partner, j + 1], R[idx, j]
                        )
                        R[partner, j + 1] = self._f_min_sum(
                            R[idx, j], L[idx, j + 1]
                        ) + R[partner, j]

            u_hat = self._hard_decision(L, R)
            if self._early_stop(u_hat, llr_ch):
                num_iters = it
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters

    def _hard_decision(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat

    def _early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, hard_ch)
