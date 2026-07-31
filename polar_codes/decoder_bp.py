"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode, bit_reversal_permutation
from _ref_function import bp_update_left, bp_update_right
from decoder_sc import _frozen_to_info_list


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        self.max_iter = max_iter
        self.alpha = alpha
        self.information_pos = _frozen_to_info_list(frozen_bits)
        self.perm = bit_reversal_permutation(N)
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.perm]
        N = self.N
        n = self.n

        left_matrix = np.zeros((N, n + 1), dtype=np.float64)
        right_matrix = np.zeros((N, n + 1), dtype=np.float64)
        left_matrix[:, n] = llr_ch
        right_matrix[:, 0] = np.array(
            [
                self.large if i not in self.information_pos else 0.0
                for i in range(N)
            ]
        )

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for i in range(n):
                left_matrix[:, n - i - 1] = bp_update_left(
                    left_matrix[:, n - i], right_matrix[:, n - i - 1], n - i
                )
            for i in range(n):
                right_matrix[:, i + 1] = bp_update_right(
                    left_matrix[:, i + 1], right_matrix[:, i], i + 1
                )

            total = left_matrix[:, 0] + right_matrix[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[total < 0] = 1
            u_hat[np.asarray(self.frozen_bits, dtype=bool) if self.frozen_bits.dtype == bool else self.frozen_bits == 1] = 0

            frozen_mask = np.ones(N, dtype=bool)
            frozen_mask[self.information_pos] = False
            u_hat[frozen_mask] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            num_iters = it
            if np.array_equal(x_hat, hard_ch):
                break

        total = left_matrix[:, 0] + right_matrix[:, 0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[total < 0] = 1
        frozen_mask = np.ones(N, dtype=bool)
        frozen_mask[self.information_pos] = False
        u_hat[frozen_mask] = 0
        return u_hat, num_iters
