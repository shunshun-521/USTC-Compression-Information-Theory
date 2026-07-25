"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from channel import hard_decision_llr


def bp_f(x, y, alpha=0.9375):
    """min-sum 近似的 f 运算。"""
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.br = bit_reversal_permutation(N)
        self.LARGE = 1e6

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = llr_ch[self.br]
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for stage in range(n - 1, -1, -1):
                span = 2 ** stage
                for i in range(0, N, 2 * span):
                    for k in range(span):
                        i0 = i + k
                        i1 = i + k + span
                        L[i0, stage] = bp_f(
                            R[i0, stage + 1] + L[i1, stage + 1],
                            L[i0, stage + 1],
                            self.alpha,
                        )
                        L[i1, stage] = (
                            bp_f(R[i0, stage + 1], L[i0, stage + 1], self.alpha)
                            + L[i1, stage + 1]
                        )

            for stage in range(n):
                span = 2 ** stage
                for i in range(0, N, 2 * span):
                    for k in range(span):
                        i0 = i + k
                        i1 = i + k + span
                        R[i0, stage + 1] = bp_f(
                            R[i1, stage + 1] + L[i1, stage + 1],
                            R[i0, stage],
                            self.alpha,
                        )
                        R[i1, stage + 1] = (
                            bp_f(R[i0, stage], L[i0, stage + 1], self.alpha)
                            + R[i1, stage + 1]
                        )

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            x_hard_br = hard_decision_llr(llr_ch)
            x_hard = np.zeros(N, dtype=int)
            x_hard[self.br] = x_hard_br
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
