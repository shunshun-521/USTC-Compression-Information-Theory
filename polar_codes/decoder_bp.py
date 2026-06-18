"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import f_operation
from encoder import bit_reversal_permutation, polar_encode


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.brev = bit_reversal_permutation(N)
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.large = 1e6

    def _minsum_f(self, a, b):
        return self.alpha * f_operation(a, b)

    def _update_left(self, L, R):
        N = self.N
        for s in range(self.n - 1, -1, -1):
            step = 1 << s
            for i in range(0, N, 2 * step):
                L[i, s] = self._minsum_f(R[i, s + 1] + L[i + step, s + 1], L[i, s + 1])
                L[i + step, s] = self._minsum_f(R[i, s + 1], L[i, s + 1]) + L[i + step, s + 1]

    def _update_right(self, L, R):
        N = self.N
        for s in range(1, self.n + 1):
            step = 1 << (s - 1)
            for i in range(0, N, 2 * step):
                R[i, s] = self._minsum_f(
                    R[i + step, s] + L[i + step, s], R[i, s - 1]
                )
                R[i + step, s] = self._minsum_f(R[i, s - 1], L[i, s]) + R[i + step, s]

    def _hard_decision(self, L, R):
        u_hat = np.zeros(self.N, dtype=int)
        total = L[:, 0] + R[:, 0]
        u_hat[total < 0] = 1
        u_hat[self.frozen_idx] = 0
        return u_hat

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = llr_ch[self.brev].copy()

        n = self.n
        N = self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            self._update_left(L, R)
            self._update_right(L, R)

            u_hat = self._hard_decision(L, R)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters
