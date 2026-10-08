"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from decoder_sc import f_operation
from encoder import bit_reversal_permutation, polar_encode


class BPDecoder:
    """BP 译码器（因子图 n+1 列）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_orig = llr_ch.copy()
        llr_ch = llr_ch[bit_reversal_permutation(self.N)]
        n = self.n
        N = self.N

        # L[i][j], R[i][j]: i in 0..N-1, j in 0..n
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=np.int32)

        for it in range(1, self.max_iter + 1):
            # Right to left: update L at columns n-1 .. 0
            for j in range(n - 1, -1, -1):
                s = 2 ** j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        i0 = i + k
                        i1 = i + k + s
                        L[i0, j] = self._f_ms(
                            R[i0, j] + L[i1, j + 1], L[i0, j + 1]
                        )
                        L[i1, j] = self._f_ms(R[i0, j], L[i0, j + 1]) + L[i1, j + 1]

            # Left to right: update R at columns 0 .. n-1
            for j in range(0, n):
                s = 2 ** j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        i0 = i + k
                        i1 = i + k + s
                        R[i0, j + 1] = self._f_ms(
                            R[i1, j] + L[i1, j + 1], R[i0, j]
                        )
                        R[i1, j + 1] = self._f_ms(R[i0, j], L[i0, j + 1]) + R[i1, j]

            # Early stopping
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_orig < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
