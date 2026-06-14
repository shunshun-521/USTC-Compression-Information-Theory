"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _minsum_f(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * min(abs(a), abs(b))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]
        self.large = 1e6

    def _hard_bits_from_llr(self, llr_ch):
        return (llr_ch < 0).astype(int)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_nat = llr_ch[self.br]

        L = np.zeros((self.N, self.n + 1), dtype=np.float64)
        R = np.zeros((self.N, self.n + 1), dtype=np.float64)
        L[:, self.n] = llr_nat
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.large

        num_iters = 0
        u_hat = np.zeros(self.N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(self.n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, self.N, step * 2):
                    for k in range(step):
                        i0 = i + k
                        i1 = i + k + step
                        L[i0, j - 1] = _minsum_f(
                            R[i0, j] + L[i1, j], L[i0, j], self.alpha
                        )
                        L[i1, j - 1] = _minsum_f(R[i0, j], L[i0, j], self.alpha) + L[
                            i1, j
                        ]

            for j in range(0, self.n):
                step = 1 << j
                for i in range(0, self.N, step * 2):
                    for k in range(step):
                        i0 = i + k
                        i1 = i + k + step
                        R[i0, j + 1] = _minsum_f(
                            R[i1, j] + L[i1, j + 1], R[i0, j], self.alpha
                        )
                        R[i1, j + 1] = (
                            _minsum_f(R[i0, j], L[i0, j + 1], self.alpha) + R[i1, j]
                        )

            num_iters = it
            for i in range(self.N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            x_hard = self._hard_bits_from_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                break

        for i in range(self.N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
