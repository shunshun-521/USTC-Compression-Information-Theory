"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from channel import hard_decision_llr
from encoder import polar_encode, bit_reversal_permutation


def _f_ms(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（列 0：信源端，列 n：信道端）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_nat = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        llr_ch = llr_nat[self.br]

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        top = i + k
                        bot = top + s
                        L[top, j - 1] = _f_ms(
                            R[top, j] + L[bot, j], L[top, j], self.alpha
                        )
                        L[bot, j - 1] = _f_ms(R[top, j], L[top, j], self.alpha) + L[bot, j]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        top = i + k
                        bot = top + s
                        R[top, j + 1] = _f_ms(
                            R[bot, j] + L[bot, j + 1], R[top, j], self.alpha
                        )
                        R[bot, j + 1] = _f_ms(R[top, j], L[top, j + 1], self.alpha) + R[bot, j]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_decision_llr(llr_nat)):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
