"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation

LARGE = 1e6


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.brp = bit_reversal_permutation(N)

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def _bp_core(self, llr):
        N = self.N
        n = self.n
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = LARGE

        u_hat = np.zeros(N, dtype=int)
        num_iters = 0

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for j in range(n - 1, -1, -1):
                s = 2 ** j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        idx2 = idx + s
                        L[idx, j] = self._f_min_sum(
                            R[idx, j + 1] + L[idx2, j + 1], L[idx, j + 1]
                        )
                        L[idx2, j] = (
                            self._f_min_sum(R[idx, j + 1], L[idx, j + 1])
                            + L[idx2, j + 1]
                        )

            for j in range(0, n):
                s = 2 ** j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        idx2 = idx + s
                        R[idx, j + 1] = self._f_min_sum(
                            R[idx2, j] + L[idx2, j + 1], R[idx, j]
                        )
                        R[idx2, j + 1] = (
                            self._f_min_sum(R[idx, j], L[idx, j + 1]) + R[idx2, j]
                        )

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_br = llr_ch[self.brp]
        u_br, num_iters = self._bp_core(llr_br)
        u_hat = u_br[self.brp]
        return u_hat, num_iters
