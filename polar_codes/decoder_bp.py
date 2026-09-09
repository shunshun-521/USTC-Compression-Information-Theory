"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode, bit_reversal_permutation


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.max_iter = max_iter
        self.alpha = alpha
        frozen = np.asarray(frozen_bits)
        if frozen.dtype != bool:
            frozen = frozen.astype(bool)
        self.frozen = frozen
        self._rev = bit_reversal_permutation(N)
        self._large = 1e6

    @staticmethod
    def _f_ms(a, b, alpha):
        return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        rev = self._rev

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch[rev]
        R[:, 0] = 0.0
        R[rev[self.frozen], 0] = self._large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, step * 2):
                    for t in range(step):
                        a = i + t
                        b = i + t + step
                        L[a, j - 1] = self._f_ms(R[a, j] + L[b, j], L[a, j], self.alpha)
                        L[b, j - 1] = self._f_ms(R[a, j], L[a, j], self.alpha) + L[b, j]

            for j in range(1, n + 1):
                step = 1 << (j - 1)
                for i in range(0, N, step * 2):
                    for t in range(step):
                        a = i + t
                        b = i + t + step
                        R[a, j] = self._f_ms(R[b, j] + L[b, j], R[a, j - 1], self.alpha)
                        R[b, j] = self._f_ms(R[a, j - 1], L[a, j], self.alpha) + R[b, j]

            total = L[:, 0] + R[:, 0]
            u_nat = np.zeros(N, dtype=int)
            u_nat[rev] = (total < 0).astype(int)
            u_nat[self.frozen] = 0

            x_hat = polar_encode(u_nat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                u_hat = u_nat
                num_iters = it
                break

            u_hat = u_nat
        else:
            num_iters = self.max_iter

        return u_hat, num_iters
