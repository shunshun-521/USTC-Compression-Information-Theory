"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode, build_generator_matrix


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6
        self.G = build_generator_matrix(N)

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def _bp_update_left(self, left_col, right_col, stage):
        N = self.N
        out = np.zeros(N)
        block = 2 ** stage
        half = block // 2
        for i in range(0, N, block):
            for k in range(half):
                i0 = i + k
                i1 = i + k + half
                out[i0] = self._f_min_sum(left_col[i1] + right_col[i1], left_col[i0])
                out[i1] = self._f_min_sum(right_col[i0], left_col[i0]) + left_col[i1]
        return out

    def _bp_update_right(self, left_col, right_col, stage):
        N = self.N
        out = np.zeros(N)
        block = 2 ** stage
        half = block // 2
        for i in range(0, N, block):
            for k in range(half):
                i0 = i + k
                i1 = i + k + half
                out[i0] = self._f_min_sum(right_col[i1] + left_col[i1], right_col[i0])
                out[i1] = self._f_min_sum(right_col[i0], left_col[i0]) + right_col[i1]
        return out

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits == 1, 0] = self.large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for stage in range(n, 0, -1):
                L[:, stage - 1] = self._bp_update_left(L[:, stage], R[:, stage - 1], stage)

            for stage in range(1, n + 1):
                R[:, stage] = self._bp_update_right(L[:, stage], R[:, stage - 1], stage)

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if self.frozen_bits[i] or total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            u_hat[i] = 0 if self.frozen_bits[i] or total >= 0 else 1

        return u_hat, num_iters
