"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode
from channel import hard_decision_llr


def bp_f(x, y, alpha):
    """min-sum f 运算。"""
    return alpha * np.sign(x) * np.sign(y) * min(abs(x), abs(y))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits).astype(bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _decide(self, L, R):
        u_hat = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            total = L[0, i] + R[0, i]
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1
        return u_hat

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        alpha = self.alpha

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n] = llr_ch.copy()
        R[0] = 0.0
        R[0, self.frozen_bits] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for s in range(n - 1, -1, -1):
                step = 1 << s
                for i in range(0, N, step * 2):
                    for j in range(step):
                        a = i + j
                        b = i + j + step
                        L[s, a] = bp_f(R[s + 1, a] + L[s + 1, b], L[s + 1, a], alpha)
                        L[s, b] = bp_f(R[s + 1, a], L[s + 1, a], alpha) + L[s + 1, b]

            for s in range(1, n + 1):
                step = 1 << (s - 1)
                for i in range(0, N, step * 2):
                    for j in range(step):
                        a = i + j
                        b = i + j + step
                        R[s, a] = bp_f(R[s - 1, b] + L[s, b], R[s - 1, a], alpha)
                        R[s, b] = bp_f(R[s - 1, a], L[s, a], alpha) + R[s - 1, b]

            u_hat = self._decide(L, R)
            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                break

        u_hat = self._decide(L, R)
        return u_hat, num_iters
