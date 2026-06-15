"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

import _sc_helpers as _fn
from decoder_sc import _frozen_to_info
from encoder import polar_encode


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        self.max_iter = max_iter
        self.alpha = alpha
        self.information_pos = _frozen_to_info(frozen_bits)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        information_pos = self.information_pos
        frozen_bit = 0

        left_matrix = np.zeros((N, n + 1), dtype=np.float64)
        right_matrix = np.zeros((N, n + 1), dtype=np.float64)
        left_matrix[:, n] = llr_ch
        temp_value = (1 - 2 * frozen_bit) * np.inf
        right_matrix[:, 0] = np.array(
            [temp_value if i not in information_pos else 0.0 for i in range(N)],
            dtype=np.float64,
        )

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for i in range(n):
                left_matrix[:, n - i - 1] = _fn.bp_update_left(
                    left_matrix[:, n - i], right_matrix[:, n - i - 1], n - i
                )
            for i in range(n):
                right_matrix[:, i + 1] = _fn.bp_update_right(
                    left_matrix[:, i + 1], right_matrix[:, i], i + 1
                )

            u_d_llr = left_matrix[:, 0] + right_matrix[:, 0]
            u_d = np.array([0 if u_d_llr[i] >= 0 else 1 for i in range(N)], dtype=int)
            u_d[information_pos] = u_d[information_pos]
            for i in range(N):
                if i not in information_pos:
                    u_d[i] = frozen_bit

            x_d_llr = left_matrix[:, n] + right_matrix[:, n]
            x_d = np.array([0 if x_d_llr[i] >= 0 else 1 for i in range(N)], dtype=int)
            x_g = polar_encode(u_d)
            if np.array_equal(x_g, x_d):
                num_iters = it
                break

        u_hat = u_d.copy()
        for i in range(N):
            if i not in information_pos:
                u_hat[i] = frozen_bit

        return u_hat, num_iters
