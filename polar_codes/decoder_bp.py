"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from channel import hard_decision_llr
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6
        self.llr_perm = bit_reversal_permutation(N)

    def _f(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch[self.llr_perm]
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 2 ** (j - 1)
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        i0 = i + t
                        i1 = i + t + s
                        L[i0, j - 1] = self._f(
                            R[i0, j] + L[i1, j],
                            L[i0, j],
                        )
                        L[i1, j - 1] = self._f(
                            R[i0, j],
                            L[i0, j],
                        ) + L[i1, j]

            for j in range(1, n + 1):
                s = 2 ** (j - 1)
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        i0 = i + t
                        i1 = i + t + s
                        R[i0, j] = self._f(
                            R[i1, j] + L[i1, j],
                            R[i0, j - 1],
                        )
                        R[i1, j] = self._f(
                            R[i0, j - 1],
                            L[i0, j],
                        ) + R[i1, j]

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    total = L[i, 0] + R[i, 0]
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            if it > 1 and np.array_equal(x_hat, hard_decision_llr(llr_ch)):
                num_iters = it
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
