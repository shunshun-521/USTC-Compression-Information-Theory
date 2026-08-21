"""
极化码 BP（置信传播）译码器
基于极化码因子图，min-sum 近似，含早停机制
"""
import numpy as np
import math

from encoder import polar_encode, bit_reversal_permutation


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """极化码因子图 BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        llr_in = llr_ch[self.br]
        L[:, n] = llr_in
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for iteration in range(self.max_iter):
            num_iters = iteration + 1

            # 从右到左更新 L
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        L[idx, j - 1] = _f_min_sum(
                            R[idx, j - 1] + L[idx + s, j],
                            L[idx, j],
                            self.alpha,
                        )
                        L[idx + s, j - 1] = (
                            _f_min_sum(R[idx, j - 1], L[idx, j], self.alpha)
                            + L[idx + s, j]
                        )

            # 从左到右更新 R
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        R[idx, j + 1] = _f_min_sum(
                            R[idx + s, j] + L[idx + s, j + 1],
                            R[idx, j],
                            self.alpha,
                        )
                        R[idx + s, j + 1] = (
                            _f_min_sum(R[idx, j], L[idx, j + 1], self.alpha)
                            + R[idx + s, j]
                        )

            total_llr = L[:, 0] + R[:, 0]
            u_hat = np.where(total_llr >= 0, 0, 1)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        total_llr = L[:, 0] + R[:, 0]
        u_hat = np.where(total_llr >= 0, 0, 1)
        u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
