"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.br = bit_reversal_permutation(N)
        self.LARGE = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        llr_internal = llr_ch[self.br]

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_internal
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for c in range(n - 1, -1, -1):
                step = 1 << c
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        i0 = i + k
                        i1 = i + k + step
                        L[i0, c] = _f_min_sum(
                            R[i0, c] + L[i1, c + 1], L[i0, c + 1], self.alpha
                        )
                        L[i1, c] = _f_min_sum(
                            R[i0, c], L[i0, c + 1], self.alpha
                        ) + L[i1, c + 1]

            for c in range(n):
                step = 1 << c
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        i0 = i + k
                        i1 = i + k + step
                        R[i0, c + 1] = _f_min_sum(
                            R[i1, c] + L[i1, c + 1], R[i0, c], self.alpha
                        )
                        R[i1, c + 1] = _f_min_sum(
                            R[i0, c], L[i0, c + 1], self.alpha
                        ) + R[i1, c]

            total_llr = L[:, 0] + R[:, 0]
            u_hat = (total_llr < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
            num_iters = it

        total_llr = L[:, 0] + R[:, 0]
        u_hat = (total_llr < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
