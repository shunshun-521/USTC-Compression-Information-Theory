"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from channel import hard_decision_llr

LARGE = 1e6


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits).astype(bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch[self.br]
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    R_top = R[i, j - 1]
                    R_bot = R[i + s, j - 1]
                    L_top = L[i, j]
                    L_bot = L[i + s, j]

                    L[i, j - 1] = _f_min_sum(R_top, L_bot + R_bot, self.alpha)
                    L[i + s, j - 1] = _f_min_sum(L_top, L_bot + R_bot, self.alpha)

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    R_top_next = R[i, j + 1]
                    R_bot = R[i + s, j + 1]
                    L_top = L[i, j + 1]
                    L_bot = L[i + s, j + 1]

                    R[i, j] = _f_min_sum(R_bot + L_bot, R_top_next, self.alpha)
                    R[i + s, j] = _f_min_sum(R_top_next, L_top, self.alpha) + R_bot

            u_br = np.zeros(N, dtype=np.int8)
            for i in range(N):
                if self.frozen_bits[i]:
                    u_br[i] = 0
                else:
                    total = L[i, 0] + R[i, 0]
                    u_br[i] = 0 if total >= 0 else 1

            u_hat = np.zeros(N, dtype=np.int8)
            u_hat[self.br] = u_br

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                break

        u_br = np.zeros(N, dtype=np.int8)
        for i in range(N):
            if self.frozen_bits[i]:
                u_br[i] = 0
            else:
                total = L[i, 0] + R[i, 0]
                u_br[i] = 0 if total >= 0 else 1
        u_hat = np.zeros(N, dtype=np.int8)
        u_hat[self.br] = u_br
        return u_hat, num_iters
