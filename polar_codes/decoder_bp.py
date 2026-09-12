"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode, bit_reversal_permutation


def _minsum(a, b, alpha=0.9375):
    return alpha * np.sign(a) * np.sign(b) * min(abs(a), abs(b))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6
        self.br = bit_reversal_permutation(N)

    def _hard_decision(self, L, R):
        total = np.array(L) + np.array(R)
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat

    def decode(self, llr_ch):
        llr_natural = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = llr_natural[self.br]
        n = self.n
        N = self.N

        L = [[0.0] * (n + 1) for _ in range(N)]
        R = [[0.0] * (n + 1) for _ in range(N)]

        for i in range(N):
            L[i][n] = llr_ch[i]
            R[i][0] = self.large if self.frozen_bits[i] else 0.0

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(n - 1, -1, -1):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    for t in range(step):
                        idx = i + t
                        s = idx + step
                        L[idx][j] = _minsum(
                            R[idx][j + 1] + L[s][j + 1], L[idx][j + 1], self.alpha
                        )
                        L[s][j] = _minsum(R[idx][j + 1], L[idx][j + 1], self.alpha) + L[s][j + 1]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    for t in range(step):
                        idx = i + t
                        s = idx + step
                        R[idx][j + 1] = _minsum(
                            R[s][j + 1] + L[s][j + 1], R[idx][j], self.alpha
                        )
                        R[s][j + 1] = _minsum(R[idx][j], L[idx][j + 1], self.alpha) + R[s][j]

            u_hat = self._hard_decision([L[i][0] for i in range(N)], [R[i][0] for i in range(N)])
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_natural < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        u_hat = self._hard_decision([L[i][0] for i in range(N)], [R[i][0] for i in range(N)])
        return u_hat, num_iters
