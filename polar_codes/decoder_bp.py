"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from decoder_sc import f_operation, sc_decode
from encoder import bit_reversal_permutation, polar_encode


def _ms_f(a, b, alpha):
    return alpha * f_operation(a, b)


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        def hard_bits(Lm, Rm):
            total = Lm[:, 0] + Rm[:, 0]
            u = (total < 0).astype(int)
            u[self.frozen_bits] = 0
            return u

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it
            L[:, n] = llr_ch
            R[:, 0] = 0.0
            R[self.frozen_bits, 0] = self.large

            for ell in range(n - 1, -1, -1):
                s = 1 << ell
                for i in range(0, N, 2 * s):
                    L[i, ell] = _ms_f(R[i, ell] + L[i + s, ell + 1], L[i, ell + 1], self.alpha)
                    L[i + s, ell] = _ms_f(R[i, ell], L[i, ell + 1], self.alpha) + L[i + s, ell + 1]

            for ell in range(0, n):
                s = 1 << ell
                for i in range(0, N, 2 * s):
                    R[i, ell + 1] = _ms_f(R[i + s, ell] + L[i + s, ell + 1], R[i, ell], self.alpha)
                    R[i + s, ell + 1] = _ms_f(R[i, ell], L[i + s, ell + 1], self.alpha) + R[i + s, ell]

            u_hat = hard_bits(L, R)
            br = bit_reversal_permutation(N)
            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat[br], x_hard):
                break

        u_hat = hard_bits(L, R)
        br = bit_reversal_permutation(N)
        if not np.array_equal(polar_encode(u_hat)[br], (llr_ch < 0).astype(int)):
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            num_iters = 0
        return u_hat, num_iters
