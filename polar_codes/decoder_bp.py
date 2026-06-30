"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self.information_pos = np.where(self.frozen_bits == 0)[0]
        self.LARGE = 1e6

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def _bp_update_left(self, left_col, right_col, layer):
        N = self.N
        interval = 2 ** (layer - 1)
        num = N // (interval * 2)
        value = np.zeros(N)
        for i in range(num):
            for j in range(interval):
                base = 2 * i * interval + j
                l0, l1 = left_col[base], left_col[base + interval]
                r0, r1 = right_col[base], right_col[base + interval]
                value[base] = self._f_min_sum(r1 + l1, l0)
                value[base + interval] = self._f_min_sum(l0, r0) + l1
        return value

    def _bp_update_right(self, left_col, right_col, layer):
        N = self.N
        interval = 2 ** (layer - 1)
        num = N // (interval * 2)
        value = np.zeros(N)
        for i in range(num):
            for j in range(interval):
                base = 2 * i * interval + j
                l0, l1 = left_col[base], left_col[base + interval]
                r0, r1 = right_col[base], right_col[base + interval]
                value[base] = self._f_min_sum(r1 + l1, r0)
                value[base + interval] = self._f_min_sum(l0, r0) + r1
        return value

    def decode(self, llr_ch):
        """主译码函数。返回：u_hat, num_iters"""
        n, N = self.n, self.N
        llr_br = np.asarray(llr_ch, dtype=np.float64)[self.br]

        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))
        L[:, n] = llr_br
        R[:, 0] = 0.0
        for i in range(N):
            if i not in self.information_pos:
                R[i, 0] = self.LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for layer in range(n, 0, -1):
                L[:, layer - 1] = self._bp_update_left(L[:, layer], R[:, layer - 1], layer)

            for layer in range(1, n + 1):
                R[:, layer] = self._bp_update_right(L[:, layer], R[:, layer - 1], layer)

            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if self.frozen_bits[i] or total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            total = L[i, 0] + R[i, 0]
            u_hat[i] = 0 if self.frozen_bits[i] or total >= 0 else 1

        return u_hat, num_iters
