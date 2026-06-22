"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode, prepare_channel_llr

LARGE = 1e6


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def _f_min_sum(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr = prepare_channel_llr(llr_ch)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    s = step
                    L[i, j - 1] = self._f_min_sum(R[i, j] + L[i + s, j], L[i, j])
                    L[i + s, j - 1] = self._f_min_sum(R[i, j], L[i, j]) + L[i + s, j]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    s = step
                    R[i, j + 1] = self._f_min_sum(
                        R[i + s, j] + L[i + s, j + 1], R[i, j]
                    )
                    R[i + s, j + 1] = self._f_min_sum(R[i, j], L[i, j + 1]) + R[i + s, j]

            for i in range(N):
                u_hat[i] = 0 if self.frozen_bits[i] else (0 if (L[i, 0] + R[i, 0]) >= 0 else 1)

            x_hat = polar_encode(u_hat)
            hard_ch = (llr < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
