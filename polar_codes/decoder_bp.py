"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import polar_sc_core as sc_core
from encoder import polar_encode


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.information_pos = list(np.where(~self.frozen_bits)[0])
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        left_matrix = np.zeros((N, n + 1))
        right_matrix = np.zeros((N, n + 1))
        left_matrix[:, n] = llr_ch
        right_matrix[:, 0] = 0.0
        right_matrix[~np.isin(np.arange(N), self.information_pos), 0] = self.LARGE

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for i in range(n):
                left_matrix[:, n - i - 1] = sc_core.bp_update_left(
                    left_matrix[:, n - i], right_matrix[:, n - i - 1], n - i
                )
            for i in range(n):
                right_matrix[:, i + 1] = sc_core.bp_update_right(
                    left_matrix[:, i + 1], right_matrix[:, i], i + 1
                )

            u_d_llr = left_matrix[:, 0] + right_matrix[:, 0]
            u_hat = (u_d_llr < 0).astype(np.int32)
            u_hat[self.frozen_bits] = 0

            x_d_llr = left_matrix[:, n] + right_matrix[:, n]
            x_d = (x_d_llr < 0).astype(np.int32)
            x_g = polar_encode(u_hat)
            if np.array_equal(x_g, x_d):
                num_iters = it
                break

        u_d_llr = left_matrix[:, 0] + right_matrix[:, 0]
        u_hat = (u_d_llr < 0).astype(np.int32)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
