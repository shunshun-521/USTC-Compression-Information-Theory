"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode, bit_reversal_permutation
from channel import hard_decision_llr


def _bp_f(x, y, alpha):
    """min-sum f 运算，带修正因子 alpha"""
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_perm = llr_ch[self.br]

        L = np.zeros((self.N, self.n + 1), dtype=np.float64)
        R = np.zeros((self.N, self.n + 1), dtype=np.float64)

        L[:, self.n] = llr_perm
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(self.N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for j in range(self.n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, self.N, 2 * step):
                    for k in range(step):
                        i1 = i + k
                        i2 = i + k + step
                        L[i1, j - 1] = _bp_f(R[i1, j] + L[i2, j], L[i1, j], self.alpha)
                        L[i2, j - 1] = _bp_f(R[i1, j], L[i1, j], self.alpha) + L[i2, j]

            for j in range(0, self.n):
                step = 1 << j
                for i in range(0, self.N, 2 * step):
                    for k in range(step):
                        i1 = i + k
                        i2 = i + k + step
                        R[i1, j + 1] = _bp_f(R[i2, j] + L[i2, j + 1], R[i1, j], self.alpha)
                        R[i2, j + 1] = _bp_f(R[i1, j], L[i1, j + 1], self.alpha) + R[i2, j]

            for i in range(self.N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_decision_llr(llr_ch)):
                break

        for i in range(self.N):
            total = L[i, 0] + R[i, 0]
            u_hat[i] = 0 if total >= 0 else 1
        u_hat[self.frozen_bits] = 0

        return u_hat, num_iters
