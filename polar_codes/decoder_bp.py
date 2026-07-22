"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation, _prepare_llr


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._rev = bit_reversal_permutation(N)

    def _minsum(self, a, b):
        return self.alpha * f_operation(a, b)

    def _hard_decision(self, L, R):
        u_hat = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
        return u_hat

    def decode(self, llr_ch):
        llr_ch = _prepare_llr(llr_ch, self.N)
        N = self.N
        n = self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it

            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        li = i + k
                        li_s = i + k + s
                        L[li, j - 1] = self._minsum(
                            R[li, j] + L[li_s, j + 1],
                            L[li, j + 1],
                        )
                        L[li_s, j - 1] = self._minsum(
                            R[li, j],
                            L[li, j + 1],
                        ) + L[li_s, j + 1]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        li = i + k
                        li_s = i + k + s
                        R[li, j + 1] = self._minsum(
                            R[li_s, j] + L[li_s, j + 1],
                            R[li, j],
                        )
                        R[li_s, j + 1] = self._minsum(
                            R[li, j],
                            L[li, j + 1],
                        ) + R[li_s, j]

            u_hat = self._hard_decision(L, R)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            hard_ch_unperm = hard_ch[self._rev]
            if np.array_equal(x_hat, hard_ch_unperm):
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters
