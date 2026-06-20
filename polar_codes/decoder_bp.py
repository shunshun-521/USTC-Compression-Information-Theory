"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（分层因子图，min-sum 近似）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def _update_left(self, L, R, layer):
        """从右向左更新 L 消息（layer 为当前列索引，更新 layer-1）。"""
        N = self.N
        interval = 2 ** (layer - 1)
        num = N // (interval * 2)
        for i in range(num):
            for j in range(interval):
                idx = 2 * i * interval + j
                l0, l1 = L[idx, layer], L[idx + interval, layer]
                r0, r1 = R[idx, layer], R[idx + interval, layer]
                L[idx, layer - 1] = self._f_min_sum(r0 + l1, l0)
                L[idx + interval, layer - 1] = self._f_min_sum(l0, r0) + l1

    def _update_right(self, L, R, layer):
        """从左向右更新 R 消息（layer 为当前列索引，更新 layer+1）。"""
        N = self.N
        interval = 2 ** layer
        num = N // (interval * 2)
        for i in range(num):
            for j in range(interval):
                idx = 2 * i * interval + j
                l0, l1 = L[idx, layer + 1], L[idx + interval, layer + 1]
                r0, r1 = R[idx, layer], R[idx + interval, layer]
                R[idx, layer + 1] = self._f_min_sum(r1 + l1, r0)
                R[idx + interval, layer + 1] = self._f_min_sum(r0, l0) + r1

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        large = 1e6

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = large

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for layer in range(n, 0, -1):
                self._update_left(L, R, layer)

            for layer in range(0, n):
                self._update_right(L, R, layer)

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
