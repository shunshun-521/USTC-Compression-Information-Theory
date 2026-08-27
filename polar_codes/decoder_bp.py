"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import f_operation
from encoder import bit_reversal_permutation, polar_encode


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
        self.rev = bit_reversal_permutation(N)

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 u_hat, num_iters
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        LARGE = 1e6

        llr_perm = llr_ch[self.rev]

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_perm
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # 右到左更新 L
            for stage in range(n, 0, -1):
                s = 1 << (stage - 1)
                for i in range(0, N, 2 * s):
                    for j in range(s):
                        idx0 = i + j
                        idx1 = i + j + s
                        L[idx0, stage - 1] = self._f_min_sum(
                            R[idx0, stage - 1] + L[idx1, stage],
                            L[idx0, stage],
                        )
                        L[idx1, stage - 1] = (
                            self._f_min_sum(R[idx0, stage - 1], L[idx0, stage])
                            + L[idx1, stage]
                        )

            # 左到右更新 R
            for stage in range(0, n):
                s = 1 << stage
                for i in range(0, N, 2 * s):
                    for j in range(s):
                        idx0 = i + j
                        idx1 = i + j + s
                        R[idx0, stage + 1] = self._f_min_sum(
                            R[idx1, stage + 1] + L[idx1, stage + 1],
                            R[idx0, stage],
                        )
                        R[idx1, stage + 1] = (
                            self._f_min_sum(R[idx0, stage], L[idx0, stage + 1])
                            + R[idx1, stage]
                        )

            # 判决与早停
            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
