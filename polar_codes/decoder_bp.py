"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器（层化因子图实现）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.LARGE = 1e6
        self._build_masks()

    def _build_masks(self):
        """为每层构建处理单元索引掩码。"""
        n, N = self.n, self.N
        self.masks = []
        for stage in range(n):
            step = 1 << stage
            mask = []
            for i in range(0, N, 2 * step):
                mask.append(i)
            self.masks.append(np.array(mask, dtype=int))

    def _f(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        N, n = self.N, self.n
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)

        L[n] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_idx] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for stage in range(n - 1, -1, -1):
                step = 1 << stage
                for i in range(0, N, 2 * step):
                    j = i + step
                    L[stage, i] = self._f(
                        L[stage + 1, i],
                        L[stage + 1, j] + R[stage, j],
                    )
                    L[stage, j] = self._f(
                        R[stage, i],
                        L[stage + 1, i],
                    ) + L[stage + 1, j]

            for stage in range(n):
                step = 1 << stage
                for i in range(0, N, 2 * step):
                    j = i + step
                    R[stage + 1, i] = self._f(
                        R[stage, i],
                        L[stage + 1, j] + R[stage, j],
                    )
                    R[stage + 1, j] = self._f(
                        R[stage, i],
                        L[stage + 1, i],
                    ) + R[stage, j]

            num_iters = it

            for idx in range(N):
                u_hat[idx] = 0 if (L[0, idx] + R[0, idx]) >= 0 else 1
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                break

        for idx in range(N):
            if self.frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if (L[0, idx] + R[0, idx]) >= 0 else 1

        return u_hat, num_iters
