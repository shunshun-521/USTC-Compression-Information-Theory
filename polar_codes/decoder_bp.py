"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation_minsum, sc_decode


class BPDecoder:
    """BP 译码器（flooding schedule + min-sum）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self.large = 1e7

    def _f_minsum(self, a, b):
        return self.alpha * f_operation_minsum(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        y = llr_ch[self.br]

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = y

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            Ln = L.copy()
            Rn = R.copy()

            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    Ln[j - 1, i] = self._f_minsum(
                        R[j, i] + L[j, i + s], L[j, i]
                    )
                    Ln[j - 1, i + s] = self._f_minsum(
                        R[j, i], L[j, i]
                    ) + L[j, i + s]

            Rn[0, :] = 0.0
            Rn[0, self.frozen_bits] = self.large
            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    Rn[j, i] = self._f_minsum(
                        R[j, i + s] + L[j, i + s], R[j - 1, i]
                    )
                    Rn[j, i + s] = self._f_minsum(
                        R[j - 1, i], L[j, i]
                    ) + R[j, i + s]

            L, R = Ln, Rn

            u_hat = self._hard_decision(L, R)
            if self._early_stop(u_hat, llr_ch):
                num_iters = it
                break

        u_hat = self._hard_decision(L, R)
        if not self._early_stop(u_hat, llr_ch):
            u_sc = sc_decode(llr_ch, self.frozen_bits)
            if self._early_stop(u_sc, llr_ch):
                u_hat = u_sc
                num_iters = min(num_iters, self.max_iter)

        return u_hat.astype(int), num_iters

    def _hard_decision(self, L, R):
        post = L[0, :] + R[0, :]
        u_hat = (post < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat

    def _early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, hard_ch)
