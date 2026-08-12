"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.br = bit_reversal_permutation(N)

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.br]
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, 0] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int64)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for j in range(n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_i = i + k
                        idx_is = i + k + s
                        L[idx_i, j + 1] = self._f_min_sum(
                            R[idx_i, j] + L[idx_is, j], L[idx_i, j]
                        )
                        L[idx_is, j + 1] = (
                            self._f_min_sum(R[idx_i, j], L[idx_i, j]) + L[idx_is, j]
                        )

            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_i = i + k
                        idx_is = i + k + s
                        R[idx_i, j + 1] = self._f_min_sum(
                            R[idx_is, j] + L[idx_is, j + 1], R[idx_i, j]
                        )
                        R[idx_is, j + 1] = (
                            self._f_min_sum(R[idx_i, j], L[idx_i, j + 1]) + R[idx_is, j]
                        )

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, n] + R[i, n]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(np.int64)
            hard_ch_br = np.empty(N, dtype=np.int64)
            hard_ch_br[self.br] = hard_ch
            if np.array_equal(x_hat, hard_ch_br):
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, n] + R[i, n]) >= 0 else 1

        return u_hat, num_iters
