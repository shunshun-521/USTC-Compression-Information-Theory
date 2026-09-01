"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import f_operation_min_sum
from encoder import polar_encode


class BPDecoder:
    """BP 译码器（分层 min-sum，每轮重置信道 LLR）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def _f(self, a, b):
        return self.alpha * f_operation_min_sum(a, b)

    def decode(self, llr_ch):
        """主译码函数"""
        n = self.n
        N = self.N
        LARGE = 1e6
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        R[self.frozen_idx, 0] = LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            L[:, n] = llr_ch

            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    L[i, j] = self._f(R[i, j + 1] + L[i + s, j + 1], L[i, j + 1])
                    L[i + s, j] = self._f(R[i, j + 1], L[i, j + 1]) + L[i + s, j + 1]

            for j in range(n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    R[i, j + 1] = self._f(R[i + s, j + 1] + L[i + s, j + 1], R[i, j])
                    R[i + s, j + 1] = self._f(R[i, j], L[i, j + 1]) + R[i + s, j + 1]

            num_iters = it
            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

            # 阻尼：将总信息反馈到右端，改善收敛
            feedback = total - llr_ch
            L[:, n] = llr_ch + 0.5 * feedback

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
