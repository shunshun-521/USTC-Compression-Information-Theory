"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation, _prepare_frozen_set


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_set = _prepare_frozen_set(frozen_bits)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _minsum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        rev = bit_reversal_permutation(self.N)
        llr = llr_ch[rev]

        n = self.n
        N = self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr

        for idx in self.frozen_set:
            R[idx, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for t in range(step):
                        li = i + t
                        ri = i + t + step
                        L[li, j - 1] = self._minsum(
                            R[li, j] + L[ri, j], L[li, j]
                        )
                        L[ri, j - 1] = self._minsum(R[li, j], L[li, j]) + L[ri, j]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    for t in range(step):
                        li = i + t
                        ri = i + t + step
                        R[li, j + 1] = self._minsum(
                            R[ri, j] + L[ri, j + 1], R[li, j]
                        )
                        R[ri, j + 1] = self._minsum(R[li, j], L[li, j + 1]) + R[ri, j]

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                if i in self.frozen_set:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard):
                num_iters = it
                break

        return u_hat, num_iters
