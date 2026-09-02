"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import _permute_llr


def _minsum_f(a, b, alpha):
    """min-sum f 运算。"""
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.m = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        llr_orig = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = _permute_llr(llr_orig)
        N = self.N
        m = self.m
        alpha = self.alpha

        L = np.zeros((N, m + 1), dtype=np.float64)
        R = np.zeros((N, m + 1), dtype=np.float64)

        L[:, m] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(self.max_iter):
            for j in range(m - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        a = i + k
                        b = a + s
                        L[a, j] = _minsum_f(
                            R[a, j + 1] + L[b, j + 1], L[a, j + 1], alpha
                        )
                        L[b, j] = _minsum_f(R[a, j + 1], L[a, j + 1], alpha) + L[b, j + 1]

            for j in range(m):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        a = i + k
                        b = a + s
                        R[a, j + 1] = _minsum_f(
                            R[b, j + 1] + L[b, j + 1], R[a, j], alpha
                        )
                        R[b, j + 1] = _minsum_f(R[a, j], L[a, j + 1], alpha) + R[b, j + 1]

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_orig < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it + 1
                break
        else:
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
