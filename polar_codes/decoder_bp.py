"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import bit_reversal_permutation, build_generator_matrix
import polar_tree_functions as fn


class BPDecoder:
    """BP 译码器（min-sum，含编码一致性早停）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.br = bit_reversal_permutation(N)
        frozen_nat = np.asarray(frozen_bits)
        self.frozen_br = frozen_nat[self.br]
        self.info_br = np.where(self.frozen_br == 0)[0]
        self.max_iter = max_iter
        self.G = build_generator_matrix(N)

    def decode(self, llr_ch):
        llr_nat = np.asarray(llr_ch, dtype=np.float64)
        y_llr = llr_nat[self.br]
        N = self.N
        n = self.n

        left_matrix = np.zeros((N, n + 1))
        right_matrix = np.zeros((N, n + 1))
        left_matrix[:, n] = y_llr
        right_matrix[:, 0] = np.where(
            np.isin(np.arange(N), self.info_br),
            0.0,
            np.inf,
        )

        num_iters = self.max_iter
        u_d = np.zeros(N, dtype=np.int8)
        for iter_num in range(1, self.max_iter + 1):
            for i in range(n):
                left_matrix[:, n - i - 1] = fn.bp_update_left(
                    left_matrix[:, n - i], right_matrix[:, n - i - 1], n - i
                )
            for i in range(n):
                right_matrix[:, i + 1] = fn.bp_update_right(
                    left_matrix[:, i + 1], right_matrix[:, i], i + 1
                )

            llr_total = left_matrix[:, 0] + right_matrix[:, 0]
            u_d = (llr_total < 0).astype(np.int8)
            u_nat = np.zeros(N, dtype=np.int8)
            u_nat[self.br] = u_d
            x_g = (u_nat @ self.G) % 2
            x_hard = (llr_nat < 0).astype(np.int8)
            if np.array_equal(x_g, x_hard):
                num_iters = iter_num
                break

        u_nat = np.zeros(N, dtype=np.int8)
        u_nat[self.br] = u_d
        return u_nat.astype(int), num_iters
