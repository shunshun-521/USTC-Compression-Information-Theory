"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import _prepare_llr, f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器（flooding schedule，min-sum 近似）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6

    def _f_ms(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        N, n = self.N, self.n
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_tree = _prepare_llr(llr_ch)

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n] = llr_tree
        R[0, self.frozen_idx] = self._large

        num_iters = self.max_iter
        hard_x = (llr_ch < 0).astype(int)

        for it in range(1, self.max_iter + 1):
            # 左到右更新 R
            for layer in range(n):
                stride = 1 << layer
                for i in range(0, N, 2 * stride):
                    for j in range(stride):
                        idx = i + j
                        R[layer + 1, idx] = self._f_ms(
                            R[layer, idx] + L[layer + 1, idx + stride],
                            R[layer, idx + stride],
                        )
                        R[layer + 1, idx + stride] = (
                            self._f_ms(R[layer, idx], L[layer + 1, idx + stride])
                            + R[layer, idx + stride]
                        )

            # 右到左更新 L
            for layer in range(n, 0, -1):
                stride = 1 << (layer - 1)
                for i in range(0, N, 2 * stride):
                    for j in range(stride):
                        idx = i + j
                        L[layer - 1, idx] = self._f_ms(
                            L[layer, idx] + R[layer, idx + stride],
                            L[layer, idx + stride],
                        )
                        L[layer - 1, idx + stride] = (
                            self._f_ms(R[layer, idx], L[layer, idx + stride])
                            + L[layer, idx + stride]
                        )

            total = L[0] + R[0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[total < 0] = 1
            u_hat[self.frozen_idx] = 0

            if np.array_equal(polar_encode(u_hat), hard_x):
                num_iters = it
                return u_hat, num_iters

        total = L[0] + R[0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[total < 0] = 1
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
