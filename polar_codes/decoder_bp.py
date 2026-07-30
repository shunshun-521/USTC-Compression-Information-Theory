"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode, _bit_rev_indices
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
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.powers = [1 << j for j in range(self.n + 1)]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = _bit_rev_indices(self.N)
        llr_ch = llr_ch[br]
        n = self.n
        N = self.N
        LARGE = 1e6

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = self.powers[j - 1]
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        L[idx, j - 1] = ms_f(
                            R[idx, j - 1] + L[idx + s, j],
                            L[idx, j],
                            self.alpha,
                        )
                        L[idx + s, j - 1] = ms_f(
                            R[idx, j - 1],
                            L[idx, j],
                            self.alpha,
                        ) + L[idx + s, j]

            for j in range(0, n):
                s = self.powers[j]
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        R[idx, j + 1] = ms_f(
                            R[idx + s, j] + L[idx + s, j + 1],
                            R[idx, j],
                            self.alpha,
                        )
                        R[idx + s, j + 1] = ms_f(
                            R[idx, j],
                            L[idx, j + 1],
                            self.alpha,
                        ) + R[idx + s, j]

            num_iters = it
            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)[br]
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                break

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            u_hat[i] = 0 if total >= 0 else 1
        u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
