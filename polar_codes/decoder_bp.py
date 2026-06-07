"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器（因子图列 0..n）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def _bp_left(self, L_col, R_col, layer):
        """从右向左更新 L 消息"""
        N = self.N
        interval = 2 ** (self.n - layer)
        value = np.zeros(N)
        for block in range(0, N, 2 * interval):
            for j in range(interval):
                i = block + j
                l0 = L_col[i + interval]
                r0 = R_col[i]
                l1 = L_col[i]
                r1 = R_col[i + interval] if i + interval < N else 0.0
                value[i] = self._f_ms(r0 + l0, l1)
                value[i + interval] = self._f_ms(r0, l1) + l0
        return value

    def _bp_right(self, L_col, R_col, layer):
        """从左向右更新 R 消息"""
        N = self.N
        interval = 2 ** (layer - 1) if layer > 0 else 1
        if layer == 0:
            interval = 1
        value = np.zeros(N)
        for block in range(0, N, 2 * interval):
            for j in range(interval):
                i = block + j
                r0 = R_col[i + interval] if i + interval < N else 0.0
                l1 = L_col[i + interval] if i + interval < N else 0.0
                r_prev = R_col[i] if layer > 0 else 0.0
                value[i] = self._f_ms(r0 + l1, r_prev)
                value[i + interval] = self._f_ms(r_prev, L_col[i]) + r0
        return value

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                L[:, j - 1] = self._bp_left(L[:, j], R[:, j - 1], j)
            for j in range(0, n):
                R[:, j + 1] = self._bp_right(L[:, j + 1], R[:, j], j + 1)

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break

        return u_hat, num_iters
