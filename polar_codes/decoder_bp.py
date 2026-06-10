"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from channel import hard_decision_llr
from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=np.int32)
        self.max_iter = max_iter
        self.alpha = alpha

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 (u_hat, num_iters)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        # L[i, j]: 从右到左消息; R[i, j]: 从左到右消息
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch

        frozen_idx = np.where(self.frozen_bits)[0]
        R[frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int32)

        for it in range(1, self.max_iter + 1):
            # 右到左更新 L
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    L[i : i + s, j - 1] = self._f_ms(
                        R[i : i + s, j] + L[i + s : i + 2 * s, j],
                        L[i : i + s, j],
                    )
                    L[i + s : i + 2 * s, j - 1] = self._f_ms(
                        R[i : i + s, j], L[i : i + s, j]
                    ) + L[i + s : i + 2 * s, j]

            # 左到右更新 R
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    R[i : i + s, j + 1] = self._f_ms(
                        R[i + s : i + 2 * s, j] + L[i + s : i + 2 * s, j + 1],
                        R[i : i + s, j],
                    )
                    R[i + s : i + 2 * s, j + 1] = self._f_ms(
                        R[i : i + s, j], L[i : i + s, j + 1]
                    ) + R[i + s : i + 2 * s, j]

            num_iters = it

            # 早停检查
            total = L[:, 0] + R[:, 0]
            u_hat = np.where(total >= 0, 0, 1).astype(np.int32)
            u_hat[self.frozen_bits.astype(bool)] = 0

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.where(total >= 0, 0, 1).astype(np.int32)
        u_hat[self.frozen_bits.astype(bool)] = 0
        return u_hat, num_iters
