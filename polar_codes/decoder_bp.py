"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode, bit_reversal_permutation


def _f_min_sum(a, b, alpha):
    sa = np.sign(a)
    sb = np.sign(b)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return alpha * sa * sb * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_ch = llr_ch.astype(np.float64)[self.br]
        N, n = self.N, self.n
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    L[i, j - 1] = _f_min_sum(
                        R[i, j] + L[i + s, j + 1], L[i, j + 1], self.alpha
                    )
                    L[i + s, j - 1] = _f_min_sum(
                        R[i, j], L[i, j + 1], self.alpha
                    ) + L[i + s, j + 1]

            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    R[i, j] = _f_min_sum(
                        R[i + s, j] + L[i + s, j + 1], R[i, j - 1], self.alpha
                    )
                    R[i + s, j] = _f_min_sum(
                        R[i, j - 1], L[i, j + 1], self.alpha
                    ) + R[i + s, j]

            num_iters = it
            u_hat = np.where(
                self.frozen_bits, 0, np.where(L[:, 0] + R[:, 0] >= 0, 0, 1)
            ).astype(int)

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        u_hat = np.where(
            self.frozen_bits, 0, np.where(L[:, 0] + R[:, 0] >= 0, 0, 1)
        ).astype(int)
        return u_hat, num_iters
