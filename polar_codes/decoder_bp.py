"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math
from decoder_sc import f_operation
from encoder import polar_encode, bit_reversal_permutation


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.LARGE = 1e6

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = llr_ch[bit_reversal_permutation(self.N)]
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 2 ** (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        i0 = i + k
                        i1 = i + k + step
                        L[i0, j - 1] = self._f_min_sum(
                            R[i0, j] + L[i1, j],
                            L[i0, j],
                        )
                        L[i1, j - 1] = self._f_min_sum(
                            R[i0, j],
                            L[i0, j],
                        ) + L[i1, j]

            for j in range(1, n + 1):
                step = 2 ** (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        i0 = i + k
                        i1 = i + k + step
                        R[i0, j] = self._f_min_sum(
                            R[i1, j] + L[i1, j],
                            R[i0, j - 1],
                        )
                        R[i1, j] = self._f_min_sum(
                            R[i0, j - 1],
                            L[i0, j],
                        ) + R[i1, j]

            for i in range(N):
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
            num_iters = it

        for i in range(N):
            u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
        u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
