"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import _frozen_to_info_pos


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.info_pos = _frozen_to_info_pos(frozen_bits)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_bit = 0

    def _f_minsum(self, x, y):
        sx = np.sign(x)
        sy = np.sign(y)
        sx = 1 if sx == 0 else sx
        sy = 1 if sy == 0 else sy
        return self.alpha * sx * sy * min(abs(x), abs(y))

    def _element_update_left(self, left, right):
        value = np.zeros(2)
        value[0] = self._f_minsum(right[1] + left[1], left[0])
        value[1] = self._f_minsum(left[0], right[0]) + left[1]
        return value

    def _element_update_right(self, left, right):
        value = np.zeros(2)
        value[0] = self._f_minsum(right[1] + left[1], right[0])
        value[1] = self._f_minsum(left[0], right[0]) + right[1]
        return value

    def _bp_update_left(self, left_col, right_col, layer_n):
        N = left_col.size
        interval = 1 << (layer_n - 1)
        value = np.zeros(N)
        for base in range(0, N, 2 * interval):
            for j in range(interval):
                idx = base + j
                left_ele = np.array([left_col[idx], left_col[idx + interval]])
                right_ele = np.array([right_col[idx], right_col[idx + interval]])
                out = self._element_update_left(left_ele, right_ele)
                value[idx] = out[0]
                value[idx + interval] = out[1]
        return value

    def _bp_update_right(self, left_col, right_col, layer_n):
        N = left_col.size
        interval = 1 << (layer_n - 1)
        value = np.zeros(N)
        for base in range(0, N, 2 * interval):
            for j in range(interval):
                idx = base + j
                left_ele = np.array([left_col[idx], left_col[idx + interval]])
                right_ele = np.array([right_col[idx], right_col[idx + interval]])
                out = self._element_update_right(left_ele, right_ele)
                value[idx] = out[0]
                value[idx + interval] = out[1]
        return value

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        br = bit_reversal_permutation(N)
        y_llr = llr_ch[br]

        left_matrix = np.zeros((N, n + 1))
        right_matrix = np.zeros((N, n + 1))
        left_matrix[:, n] = y_llr

        temp_value = (1 - 2 * self.frozen_bit) * np.inf
        info_set = set(self.info_pos.tolist())
        right_matrix[:, 0] = np.array(
            [0.0 if i in info_set else temp_value for i in range(N)]
        )

        num_iters = 0
        for it in range(self.max_iter):
            num_iters = it + 1
            for i in range(n):
                left_matrix[:, n - i - 1] = self._bp_update_left(
                    left_matrix[:, n - i], right_matrix[:, n - i - 1], n - i
                )
            for i in range(n):
                right_matrix[:, i + 1] = self._bp_update_right(
                    left_matrix[:, i + 1], right_matrix[:, i], i + 1
                )

            u_d_llr = left_matrix[:, 0] + right_matrix[:, 0]
            u_hat = np.array([0 if u_d_llr[i] >= 0 else 1 for i in range(N)], dtype=int)
            for i in range(N):
                if i not in info_set:
                    u_hat[i] = self.frozen_bit

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                return u_hat, num_iters

        u_d_llr = left_matrix[:, 0] + right_matrix[:, 0]
        u_hat = np.array([0 if u_d_llr[i] >= 0 else 1 for i in range(N)], dtype=int)
        for i in range(N):
            if i not in info_set:
                u_hat[i] = self.frozen_bit
        return u_hat, num_iters
