"""
极化码 BP（置信传播）译码器
基于因子图，min-sum 近似，含早停
"""
import numpy as np

import ref_function as _ref_function
from decoder_sc import _frozen_to_info_pos, _llr_for_decoder
from encoder import polar_encode


class BPDecoder:
    """BP 译码器（g_matrix 早停）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.info_pos = _frozen_to_info_pos(self.frozen_bits)
        self.max_iter = int(max_iter)
        self.alpha = float(alpha)
        self._frozen_value = 0

    def decode(self, llr_ch):
        llr = _llr_for_decoder(llr_ch)
        N = self.N
        n = int(np.log2(N))

        left_matrix = np.zeros((N, n + 1))
        right_matrix = np.zeros((N, n + 1))
        left_matrix[:, n] = llr
        temp_value = (1 - 2 * self._frozen_value) * np.inf
        temp = np.array(
            [temp_value if i not in self.info_pos else 0.0 for i in range(N)],
            dtype=np.float64,
        )
        right_matrix[:, 0] = temp

        flag = 0
        iter_num = 1
        u_d = np.zeros(N, dtype=np.int8)

        while flag == 0 and iter_num <= self.max_iter:
            for i in range(n):
                left_matrix[:, n - i - 1] = _ref_function.bp_update_left(
                    left_matrix[:, n - i], right_matrix[:, n - i - 1], n - i
                )
            for i in range(n):
                right_matrix[:, i + 1] = _ref_function.bp_update_right(
                    left_matrix[:, i + 1], right_matrix[:, i], i + 1
                )

            u_d_llr = left_matrix[:, 0] + right_matrix[:, 0]
            u_d = np.array([0 if u_d_llr[i] >= 0 else 1 for i in range(N)], dtype=np.int8)
            u_d[self.frozen_bits] = 0

            x_d_llr = left_matrix[:, n] + right_matrix[:, n]
            x_d = np.array([0 if x_d_llr[i] >= 0 else 1 for i in range(N)], dtype=np.int8)
            x_hat = polar_encode(u_d)
            if np.array_equal(x_hat, x_d):
                flag = 1
            iter_num += 1

        num_iters = iter_num - 1 if flag else self.max_iter
        return u_d, num_iters
