"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math
from encoder import polar_encode
from channel import hard_decision_llr


def _f_minsum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（极化因子图，列 0..n）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        # L[i][j], R[i][j]: i in 0..N-1, j in 0..n
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self._large

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            # right to left: update L at columns n-1 .. 0
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        i0 = i + k
                        i1 = i + k + s
                        L[i0, j - 1] = _f_minsum(
                            R[i0, j] + L[i1, j], L[i0, j], self.alpha
                        )
                        L[i1, j - 1] = _f_minsum(R[i0, j], L[i0, j], self.alpha) + L[i1, j]

            # left to right: update R at columns 0 .. n-1
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        i0 = i + k
                        i1 = i + k + s
                        R[i1, j + 1] = _f_minsum(
                            R[i0, j] + L[i1, j + 1], R[i0, j + 1], self.alpha
                        ) + R[i1, j]
                        R[i0, j + 1] = _f_minsum(
                            R[i1, j] + L[i1, j + 1], R[i0, j], self.alpha
                        )

            # decision and early stop
            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(np.int8)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(np.int8)
        u_hat[self.frozen_bits] = 0
        return u_hat.astype(int), num_iters
