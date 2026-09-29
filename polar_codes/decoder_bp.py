"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode, bit_reversal_permutation


def ms_f(a, b, alpha=0.9375):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（因子图列 0..n）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        fb = np.asarray(frozen_bits)
        self.frozen_bits = fb.astype(bool) if fb.dtype == bool else (fb != 0)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        frozen = self.frozen_bits

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch[self.br]
        R[:, 0] = 0.0
        R[frozen, 0] = self.large

        num_iters = 0
        for it in range(self.max_iter):
            num_iters = it + 1
            for j in range(n, 0, -1):
                stride = 1 << (j - 1)
                for block in range(0, N, 2 * stride):
                    for i in range(stride):
                        a = block + i
                        b = a + stride
                        L[a, j - 1] = ms_f(R[a, j] + L[b, j], L[a, j], self.alpha)
                        L[b, j - 1] = ms_f(R[a, j], L[a, j], self.alpha) + L[b, j]

            for j in range(0, n):
                stride = 1 << j
                for block in range(0, N, 2 * stride):
                    for i in range(stride):
                        a = block + i
                        b = a + stride
                        R[a, j + 1] = ms_f(R[b, j] + L[b, j + 1], R[a, j], self.alpha)
                        R[b, j + 1] = ms_f(R[a, j], L[a, j + 1], self.alpha) + R[b, j]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[frozen] = 0
            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[frozen] = 0
        return u_hat, num_iters
