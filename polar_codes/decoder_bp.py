"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math
from encoder import polar_encode, bit_reversal_permutation


def _f_min_sum(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器（极化码因子图）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        br = bit_reversal_permutation(N)
        llr_nat = np.empty(N, dtype=np.float64)
        llr_nat[br] = llr_ch
        L[:, n] = llr_nat
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx0 = i + k
                        idx1 = i + k + s
                        L[idx0, j - 1] = _f_min_sum(
                            R[idx0, j] + L[idx1, j],
                            L[idx0, j],
                            self.alpha,
                        )
                        L[idx1, j - 1] = (
                            _f_min_sum(R[idx0, j], L[idx0, j], self.alpha)
                            + L[idx1, j]
                        )

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx0 = i + k
                        idx1 = i + k + s
                        R[idx0, j + 1] = _f_min_sum(
                            R[idx1, j] + L[idx1, j + 1],
                            R[idx0, j],
                            self.alpha,
                        )
                        R[idx1, j + 1] = (
                            _f_min_sum(R[idx0, j], L[idx0, j + 1], self.alpha)
                            + R[idx1, j]
                        )

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
