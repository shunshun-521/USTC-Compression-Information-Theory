"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停
"""
import math
import numpy as np

from encoder import polar_encode_core, bit_reversal_permutation


class BPDecoder:
    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.LARGE = 1e6

    def _f(self, x, y):
        return self.alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        rev = bit_reversal_permutation(self.N)
        llr_ch = llr_ch[rev]
        n, N = self.n, self.N

        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for base in range(0, N, 2 * s):
                    for k in range(s):
                        u = base + k
                        l = base + k + s
                        L[u, j - 1] = self._f(R[u, j - 1] + L[l, j], L[u, j])
                        L[l, j - 1] = self._f(R[u, j - 1], L[u, j]) + L[l, j]

            for j in range(0, n):
                s = 1 << j
                for base in range(0, N, 2 * s):
                    for k in range(s):
                        u = base + k
                        l = base + k + s
                        R[u, j + 1] = self._f(R[l, j] + L[l, j + 1], R[u, j])
                        R[l, j + 1] = self._f(R[u, j], L[u, j + 1]) + R[l, j]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_core_hat = polar_encode_core(u_hat)
            hard_x_core = (llr_ch < 0).astype(int)
            if np.array_equal(x_core_hat, hard_x_core):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
