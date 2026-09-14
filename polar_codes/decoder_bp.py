"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode, bit_reversal_permutation
from channel import hard_decision_llr


def ms_f(x, y, alpha):
    """min-sum f 运算"""
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6
        self.rev = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        """主译码函数"""
        N = self.N
        m = self.n
        L = np.zeros((m + 1, N), dtype=np.float64)
        R = np.zeros((m + 1, N), dtype=np.float64)

        L[m, :] = llr_ch.astype(np.float64)[self.rev]
        R[0, :] = 0.0
        R[0, self.frozen_bits] = self.large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for s in range(m):
                step = 1 << s
                for i in range(0, N, 2 * step):
                    for j in range(step):
                        a = i + j
                        b = i + j + step
                        R[s + 1, a] = ms_f(R[s, a] + L[s + 1, b], R[s, b], self.alpha)
                        R[s + 1, b] = ms_f(R[s, a], L[s + 1, a], self.alpha) + R[s, b]

            for s in range(m - 1, -1, -1):
                step = 1 << s
                for i in range(0, N, 2 * step):
                    for j in range(step):
                        a = i + j
                        b = i + j + step
                        L[s, a] = ms_f(L[s + 1, a] + R[s + 1, b], L[s + 1, b], self.alpha)
                        L[s, b] = ms_f(R[s + 1, a], L[s + 1, a], self.alpha) + L[s + 1, b]

            total = L[0, :] + R[0, :]
            u_hat = np.where(self.frozen_bits, 0, (total < 0).astype(int))
            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_decision_llr(llr_ch)):
                num_iters = it
                break

        total = L[0, :] + R[0, :]
        u_hat = np.where(self.frozen_bits, 0, (total < 0).astype(int))
        return u_hat, num_iters
