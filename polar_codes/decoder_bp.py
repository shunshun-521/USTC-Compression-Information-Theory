"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import bit_reversal_permutation, polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        rev = bit_reversal_permutation(N)
        L[n, :] = llr_ch[rev]
        R[0, :] = 0.0
        R[0, self.frozen_idx] = self.LARGE

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        i0, i1 = i + t, i + t + s
                        L[j - 1, i0] = _f_min_sum(R[j, i0] + L[j, i1], L[j, i0], self.alpha)
                        L[j - 1, i1] = _f_min_sum(R[j, i0], L[j, i0], self.alpha) + L[j, i1]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        i0, i1 = i + t, i + t + s
                        R[j + 1, i0] = _f_min_sum(R[j, i1] + L[j, i1], R[j, i0], self.alpha)
                        R[j + 1, i1] = _f_min_sum(R[j, i0], L[j, i0], self.alpha) + R[j, i1]

            u_hat = (L[0, :] + R[0, :] < 0).astype(int)
            u_hat[self.frozen_bits] = 0
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        u_hat = (L[0, :] + R[0, :] < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
