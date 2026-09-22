"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode
from decoder_sc import _frozen_to_bool


class BPDecoder:
    """
    BP 译码器（因子图 L[i,j] / R[i,j]，i=节点，j=阶段 0..n）。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = _frozen_to_bool(frozen_bits)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen)[0]
        self.LARGE = 1e6

    def _min_sum(self, a, b):
        return self.alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        m = self.n

        L = np.zeros((N, m + 1), dtype=np.float64)
        R = np.zeros((N, m + 1), dtype=np.float64)
        L[:, m] = llr_ch
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(self.max_iter):
            num_iters = it + 1

            for j in range(m - 1, -1, -1):
                block = 1 << (m - j)
                half = block // 2
                for i in range(0, N, block):
                    for k in range(half):
                        top = i + k
                        bot = i + k + half
                        L[bot, j] = L[bot, j + 1] + self._min_sum(L[top, j + 1], R[top, j])
                        L[top, j] = self._min_sum(L[top, j + 1], L[bot, j + 1] + R[bot, j])

            for j in range(m):
                block = 1 << (m - j)
                half = block // 2
                for i in range(0, N, block):
                    for k in range(half):
                        top = i + k
                        bot = i + k + half
                        R[top, j + 1] = self._min_sum(R[top, j], L[bot, j + 1] + R[bot, j])
                        R[bot, j + 1] = R[bot, j] + self._min_sum(L[top, j + 1], R[top, j])

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen] = 0
        return u_hat, num_iters
