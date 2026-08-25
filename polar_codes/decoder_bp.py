"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode
from _polar_ref_function import bp_update_left, bp_update_right


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        self.max_iter = max_iter
        self.alpha = alpha
        self.information_pos = list(np.where(~self.frozen_bits.astype(bool))[0])
        self.frozen_bit = 0
        self._LARGE = 1e6

    def _hard_decision(self, left_matrix, right_matrix):
        total = left_matrix[:, 0] + right_matrix[:, 0]
        u_hat = (total < 0).astype(int)
        frozen_idx = np.where(self.frozen_bits.astype(bool))[0]
        u_hat[frozen_idx] = 0
        return u_hat

    def _check_early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, hard_ch)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        left_matrix = np.zeros((N, n + 1), dtype=np.float64)
        right_matrix = np.zeros((N, n + 1), dtype=np.float64)
        left_matrix[:, n] = llr_ch
        temp_value = (1 - 2 * self.frozen_bit) * self._LARGE
        right_matrix[:, 0] = np.array(
            [temp_value if i not in self.information_pos else 0.0 for i in range(N)],
            dtype=np.float64,
        )

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it
            for i in range(n):
                left_matrix[:, n - i - 1] = bp_update_left(
                    left_matrix[:, n - i], right_matrix[:, n - i - 1], n - i
                )
            for i in range(n):
                right_matrix[:, i + 1] = bp_update_right(
                    left_matrix[:, i + 1], right_matrix[:, i], i + 1
                )

            u_hat = self._hard_decision(left_matrix, right_matrix)
            if self._check_early_stop(u_hat, llr_ch):
                break

        u_hat = self._hard_decision(left_matrix, right_matrix)
        return u_hat, num_iters
