"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode, bit_reversal_permutation


LARGE = 1e6


def _f_minsum(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器（因子图列 0=信源，列 n=信道）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        if self.frozen_bits.dtype != bool:
            self.frozen_bits = self.frozen_bits.astype(int) == 1
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)

    def _hard_decision(self, L, R):
        u = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            if self.frozen_bits[i]:
                u[i] = 0
            else:
                u[i] = 0 if (L[i][0] + R[i][0]) >= 0 else 1
        return u

    def decode(self, llr_ch):
        n = self.n
        N = self.N
        br = self.br
        llr = np.asarray(llr_ch, dtype=np.float64)[br]

        L = [[0.0] * (n + 1) for _ in range(N)]
        R = [[0.0] * (n + 1) for _ in range(N)]

        for i in range(N):
            L[i][n] = llr[i]
            R[i][0] = LARGE if self.frozen_bits[i] else 0.0

        num_iters = 0
        for it in range(self.max_iter):
            num_iters = it + 1

            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for block in range(0, N, 2 * step):
                    for i in range(block, block + step):
                        s = i + step
                        L[i][j - 1] = _f_minsum(R[i][j] + L[s][j], L[i][j], self.alpha)
                        L[s][j - 1] = _f_minsum(R[i][j], L[i][j], self.alpha) + L[s][j]

            for j in range(0, n):
                step = 1 << j
                for block in range(0, N, 2 * step):
                    for i in range(block, block + step):
                        s = i + step
                        R[i][j + 1] = _f_minsum(R[s][j] + L[s][j + 1], R[i][j], self.alpha)
                        R[s][j + 1] = _f_minsum(R[i][j], L[s][j + 1], self.alpha) + R[s][j]

            u_hat = self._hard_decision(L, R)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters
