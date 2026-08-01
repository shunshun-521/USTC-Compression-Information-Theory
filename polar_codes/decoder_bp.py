"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode, bit_reversal_permutation


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        if 2 ** self.n != N:
            raise ValueError("N must be a power of 2")
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self._large = 1e6

    def _minsum(self, a, b):
        return self.alpha * f_operation(a, b)

    def _hard_bits_from_llr(self, llr_ch):
        return (llr_ch < 0).astype(int)

    def _check_early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        br = bit_reversal_permutation(self.N)
        x_hard = self._hard_bits_from_llr(llr_ch)
        return np.array_equal(x_hat, x_hard)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64).copy()
        br = bit_reversal_permutation(self.N)
        llr_natural = llr_ch[br]

        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_natural
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self._large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    L[i, j - 1] = self._minsum(
                        R[i, j] + L[i + s, j],
                        L[i, j],
                    )
                    L[i + s, j - 1] = self._minsum(R[i, j], L[i, j]) + L[i + s, j]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    R[i, j + 1] = self._minsum(
                        R[i + s, j] + L[i + s, j + 1],
                        R[i, j],
                    )
                    R[i + s, j + 1] = self._minsum(R[i, j], L[i, j + 1]) + R[i + s, j]

            num_iters = it
            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            if self._check_early_stop(u_hat, llr_ch):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
