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
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.rev = bit_reversal_permutation(N)
        self.LARGE = 1e6

    def _checknode(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        """主译码函数。"""
        m = self.n
        N = self.N
        llr_ch_natural = np.asarray(llr_ch, dtype=np.float64)
        llr_internal = llr_ch_natural[self.rev]

        L = [np.zeros(N, dtype=np.float64) for _ in range(m + 1)]
        R = [np.zeros(N, dtype=np.float64) for _ in range(m + 1)]

        L[m][:] = llr_internal
        R[0][:] = 0.0
        R[0][self.frozen_idx] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for i in range(m - 1, -1, -1):
                step = 1 << i
                for j in range(0, N, 2 * step):
                    L[i][j:j + step] = self._checknode(
                        L[i + 1][j:j + step],
                        L[i + 1][j + step:j + 2 * step] + R[i][j + step:j + 2 * step],
                    )
                    L[i][j + step:j + 2 * step] = self._checknode(
                        R[i][j:j + step],
                        L[i + 1][j:j + step],
                    ) + L[i + 1][j + step:j + 2 * step]

            for i in range(m):
                step = 1 << i
                for j in range(0, N, 2 * step):
                    R[i + 1][j:j + step] = self._checknode(
                        R[i][j:j + step],
                        L[i + 1][j + step:j + 2 * step] + R[i][j + step:j + 2 * step],
                    )
                    R[i + 1][j + step:j + 2 * step] = self._checknode(
                        L[i + 1][j:j + step],
                        R[i][j:j + step],
                    ) + R[i][j + step:j + 2 * step]

            total = L[0] + R[0]
            u_hat = (total < 0).astype(np.int8)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch_natural < 0).astype(np.int8)
            if np.array_equal(x_hat, hard_ch):
                break

        total = L[0] + R[0]
        u_hat = (total < 0).astype(np.int8)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
