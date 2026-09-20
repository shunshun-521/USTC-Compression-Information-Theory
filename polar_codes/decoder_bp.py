"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.rev = bit_reversal_permutation(N)

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        rev = self.rev

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, 0] = llr_ch
        R[:, 0] = 0.0
        frozen_tree = self.frozen_bits
        R[frozen_tree == 1, 0] = self.LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for s in range(n):
                step = 1 << s
                for i in range(0, N, 2 * step):
                    L[i, s + 1] = self._f_min_sum(
                        R[i, s] + L[i + step, s],
                        L[i, s],
                    )
                    L[i + step, s + 1] = self._f_min_sum(
                        R[i, s],
                        L[i, s],
                    ) + L[i + step, s]

            for s in range(n - 1, -1, -1):
                step = 1 << s
                for i in range(0, N, 2 * step):
                    R[i, s + 1] = self._f_min_sum(
                        R[i + step, s] + L[i + step, s + 1],
                        R[i, s],
                    )
                    R[i + step, s + 1] = self._f_min_sum(
                        R[i, s],
                        L[i, s + 1],
                    ) + R[i + step, s]

            total = L[:, n] + R[:, n]
            u_hat = np.zeros(N, dtype=int)
            for t in range(N):
                if frozen_tree[t]:
                    u_hat[t] = 0
                else:
                    u_hat[t] = 0 if total[t] >= 0 else 1
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[:, n] + R[:, n]
        u_hat = np.zeros(N, dtype=int)
        for t in range(N):
            if frozen_tree[t]:
                u_hat[t] = 0
            else:
                u_hat[t] = 0 if total[t] >= 0 else 1

        return u_hat, num_iters
