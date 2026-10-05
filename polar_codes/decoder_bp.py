"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode, bit_reversal_permutation
from channel import hard_decision_llr

_LARGE = 1e6


class BPDecoder:
    """
    BP 译码器。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def _f_ms(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        rev = bit_reversal_permutation(N)
        llr_nat = llr_ch.copy()
        llr_ch = llr_ch[rev]

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = _LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        idx2 = i + k + s
                        L[idx, j - 1] = self._f_ms(
                            R[idx, j] + L[idx2, j], L[idx, j]
                        )
                        L[idx2, j - 1] = self._f_ms(R[idx, j], L[idx, j]) + L[idx2, j]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        idx2 = i + k + s
                        R[idx, j + 1] = self._f_ms(
                            R[idx2, j] + L[idx2, j + 1], R[idx, j]
                        )
                        R[idx2, j + 1] = self._f_ms(R[idx, j], L[idx, j + 1]) + R[idx2, j]

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_nat)
            if np.array_equal(x_hat, x_hard):
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
