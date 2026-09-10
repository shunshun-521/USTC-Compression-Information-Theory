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
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.rev = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_perm = llr_ch[self.rev]
        n = self.n
        N = self.N
        LARGE = 1e4

        left = np.zeros((n + 1, N), dtype=np.float64)
        right = np.zeros((n + 1, N), dtype=np.float64)
        left[n, :] = llr_perm
        right[0, :] = 0.0
        right[0, self.frozen_bits] = LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(self.max_iter):
            num_iters = it + 1

            for s in range(n - 1, -1, -1):
                stride = 1 << s
                for base in range(0, N, stride << 1):
                    for j in range(stride):
                        i = base + j
                        left[s, i] = _f_min_sum(
                            left[s + 1, i] + right[s + 1, i],
                            left[s + 1, i + stride] + right[s + 1, i + stride],
                            self.alpha,
                        )
                        left[s, i + stride] = (
                            _f_min_sum(
                                right[s + 1, i], left[s + 1, i], self.alpha
                            )
                            + left[s + 1, i + stride]
                        )

            for s in range(n):
                stride = 1 << s
                for base in range(0, N, stride << 1):
                    for j in range(stride):
                        i = base + j
                        right[s + 1, i] = _f_min_sum(
                            right[s, i + stride] + left[s + 1, i + stride],
                            right[s, i],
                            self.alpha,
                        )
                        right[s + 1, i + stride] = (
                            _f_min_sum(
                                right[s, i], left[s + 1, i], self.alpha
                            )
                            + right[s, i + stride]
                        )

            for i in range(N):
                total = left[0, i] + right[0, i]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            total = left[0, i] + right[0, i]
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
