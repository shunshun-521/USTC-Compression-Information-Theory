"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6
        self.br = bit_reversal_permutation(N)

    def _minsum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """主译码函数。"""
        N, n = self.N, self.n
        br = self.br
        llr = np.asarray(llr_ch, dtype=np.float64)[br]

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, 0] = llr
        R[:, n] = 0.0
        R[self.frozen_bits, n] = self.large

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for s in range(n):
                block = 1 << (s + 1)
                half = block >> 1
                for j in range(0, N, block):
                    for t in range(half):
                        idx = j + t
                        L[idx, s + 1] = self._minsum(
                            R[idx, s] + L[idx + half, s], L[idx, s]
                        )
                        L[idx + half, s + 1] = self._minsum(
                            R[idx, s], L[idx, s]
                        ) + L[idx + half, s]

            for s in range(n - 1, -1, -1):
                block = 1 << (s + 1)
                half = block >> 1
                for j in range(0, N, block):
                    for t in range(half):
                        idx = j + t
                        R[idx, s] = self._minsum(
                            R[idx + half, s + 1] + L[idx + half, s + 1], R[idx, s + 1]
                        )
                        R[idx + half, s] = self._minsum(
                            R[idx, s + 1], L[idx, s + 1]
                        ) + R[idx + half, s + 1]

            total = L[:, n] + R[:, n]
            u_hat = np.where(total >= 0, 0, 1).astype(np.int8)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard):
                num_iters = it
                break

        total = L[:, n] + R[:, n]
        u_hat = np.where(total >= 0, 0, 1).astype(np.int8)
        u_hat[self.frozen_bits] = 0
        return u_hat.astype(int), num_iters
