"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode, bit_reversal_permutation
from channel import hard_decision_llr


def _f_minsum(x, y, alpha):
    sign_x = np.sign(x)
    sign_y = np.sign(y)
    sign_x[sign_x == 0] = 1
    sign_y[sign_y == 0] = 1
    return alpha * sign_x * sign_y * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_internal = llr_ch[self.br]

        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_internal
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for j in range(n - 1, -1, -1):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    s = step
                    La = R[i: i + s, j] + L[i + s: i + 2 * s, j + 1]
                    Lb = L[i: i + s, j + 1]
                    L[i: i + s, j] = _f_minsum(La, Lb, self.alpha)
                    L[i + s: i + 2 * s, j] = (
                        _f_minsum(R[i: i + s, j], Lb, self.alpha) + L[i + s: i + 2 * s, j + 1]
                    )

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    s = step
                    Ra = R[i + s: i + 2 * s, j] + L[i + s: i + 2 * s, j + 1]
                    Rb = R[i: i + s, j]
                    Lb = L[i: i + s, j + 1]
                    R[i: i + s, j + 1] = _f_minsum(Ra, Rb, self.alpha)
                    R[i + s: i + 2 * s, j + 1] = (
                        _f_minsum(Rb, Lb, self.alpha) + R[i + s: i + 2 * s, j]
                    )

            total = L[:, 0] + R[:, 0]
            u_hat = np.where(total >= 0, 0, 1).astype(np.int8)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.where(total >= 0, 0, 1).astype(np.int8)
        u_hat[self.frozen_bits] = 0
        return u_hat.astype(int), num_iters
