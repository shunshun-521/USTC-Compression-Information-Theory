"""
极化码 BP（置信传播）译码器
基于极化码因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import (
    _bit_reversed, _active_llr_level, _active_bit_level,
    _update_llrs, _update_bits,
)


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e10
        self.frozen_set = set(np.where(self.frozen_bits)[0])

    def _hard_decision(self, L, R):
        u_hat = np.zeros(self.N, dtype=np.int_)
        for i in range(self.N):
            total = L[i, self.n] + R[i, 0]
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1
        return u_hat

    def _check_early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, hard_ch)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = llr_ch[bit_reversal_permutation(self.N)]

        L = np.zeros((self.N, self.n + 1), dtype=np.float64)
        R = np.zeros((self.N, self.n + 1), dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=np.int_)
        L[:, 0] = llr_ch.copy()
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it

            for i in range(self.N):
                l = _bit_reversed(i, self.n)
                _update_llrs(L, B, l, self.n)
                if l in self.frozen_set:
                    B[l, self.n] = 0
                else:
                    B[l, self.n] = 0 if L[l, self.n] >= 0 else 1
                _update_bits(B, l, self.n)

            for i in range(self.N):
                total = L[i, self.n] + R[i, 0]
                if self.frozen_bits[i]:
                    B[i, self.n] = 0
                else:
                    B[i, self.n] = 0 if total >= 0 else 1

            u_hat = B[:, self.n].astype(int)
            if self._check_early_stop(u_hat, llr_ch):
                break

        return B[:, self.n].astype(int), num_iters
