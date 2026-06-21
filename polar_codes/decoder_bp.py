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
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.br = bit_reversal_permutation(N)

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def _hard_decision(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat

    def _check_early_stop(self, L, R, llr_ch_orig):
        u_hat = self._hard_decision(L, R)
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_ch_orig < 0).astype(int)
        return np.array_equal(x_hat, hard_ch)

    def decode(self, llr_ch):
        """主译码函数。"""
        llr_ch_orig = np.asarray(llr_ch, dtype=np.float64)
        llr_internal = llr_ch_orig[self.br]
        n, N = self.n, self.N
        large = 1e6
        num_iters = 0

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        for _ in range(self.max_iter):
            num_iters += 1
            L[:, n] = llr_internal
            R[:, 0] = 0.0
            R[self.frozen_idx, 0] = large

            for s in range(1, n + 1):
                block = 1 << s
                half = block >> 1
                for start in range(0, N, block):
                    for k in range(half):
                        top = start + k
                        bot = start + k + half
                        R[top, s] = self._f_min_sum(
                            R[top, s - 1], L[bot, s] + R[bot, s - 1]
                        )
                        R[bot, s] = self._f_min_sum(R[top, s - 1], L[top, s]) + R[bot, s - 1]

            for s in range(n, 0, -1):
                block = 1 << s
                half = block >> 1
                for start in range(0, N, block):
                    for k in range(half):
                        top = start + k
                        bot = start + k + half
                        L[top, s - 1] = self._f_min_sum(
                            L[top, s], L[bot, s] + R[bot, s]
                        )
                        L[bot, s - 1] = self._f_min_sum(R[top, s], L[top, s]) + L[bot, s]

            if self._check_early_stop(L, R, llr_ch_orig):
                break

        return self._hard_decision(L, R), num_iters
