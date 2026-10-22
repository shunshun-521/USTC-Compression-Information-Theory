"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from channel import hard_decision_llr


class BPDecoder:
    """BP 译码器（极化码因子图，列 0 为信源端，列 n 为信道端）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._rev = bit_reversal_permutation(N)
        self._inv = np.empty(N, dtype=int)
        self._inv[self._rev] = np.arange(N)

    def _f_min_sum(self, a, b):
        return self.alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_dec = llr_ch[self._inv]

        L = np.zeros((self.N, self.n + 1), dtype=np.float64)
        R = np.zeros((self.N, self.n + 1), dtype=np.float64)
        L[:, self.n] = llr_dec
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = 1e6

        num_iters = 0
        u_hat = np.zeros(self.N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            for j in range(self.n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, self.N, 2 * step):
                    for k in range(step):
                        i1 = i + k
                        i2 = i + k + step
                        L[i1, j - 1] = self._f_min_sum(
                            R[i1, j] + L[i2, j], L[i1, j]
                        )
                        L[i2, j - 1] = self._f_min_sum(R[i1, j], L[i1, j]) + L[i2, j]

            for j in range(0, self.n):
                step = 1 << j
                for i in range(0, self.N, 2 * step):
                    for k in range(step):
                        i1 = i + k
                        i2 = i + k + step
                        R[i1, j + 1] = self._f_min_sum(
                            R[i2, j] + L[i2, j + 1], R[i1, j]
                        )
                        R[i2, j + 1] = self._f_min_sum(R[i1, j], L[i1, j + 1]) + R[i2, j]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
