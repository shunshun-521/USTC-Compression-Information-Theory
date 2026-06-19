"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from decoder_sc import f_operation, bit_reversal_permutation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.br = bit_reversal_permutation(N)

    def _minsum_f(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_nat = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = llr_nat[self.br]

        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 2 ** (j - 1)
                for block in range(0, N, 2 * s):
                    for i in range(block, block + s):
                        L[i, j - 1] = self._minsum_f(
                            R[i, j] + L[i + s, j], L[i, j]
                        )
                        L[i + s, j - 1] = self._minsum_f(R[i, j], L[i, j]) + L[
                            i + s, j
                        ]

            for j in range(0, n):
                s = 2 ** j
                for block in range(0, N, 2 * s):
                    for i in range(block, block + s):
                        R[i, j + 1] = self._minsum_f(
                            R[i + s, j] + L[i + s, j + 1], R[i, j]
                        )
                        R[i + s, j + 1] = self._minsum_f(R[i, j], L[i, j + 1]) + R[
                            i + s, j
                        ]

            num_iters = it
            for i in range(N):
                u_hat[i] = 0 if self.frozen_bits[i] else (
                    0 if (L[i, 0] + R[i, 0]) >= 0 else 1
                )

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_nat < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            u_hat[i] = 0 if self.frozen_bits[i] else (
                0 if (L[i, 0] + R[i, 0]) >= 0 else 1
            )

        return u_hat, num_iters
