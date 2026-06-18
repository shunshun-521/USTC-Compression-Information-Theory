"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import _frozen_to_info_indices
import sc_core as scf


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        self.info_indices = _frozen_to_info_indices(frozen_bits)
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        N = self.N
        n = self.n
        br = bit_reversal_permutation(N)
        llr = np.asarray(llr_ch, dtype=np.float64)[br]

        left_matrix = np.zeros((N, n + 1))
        right_matrix = np.zeros((N, n + 1))
        left_matrix[:, n] = llr
        temp = np.zeros(N)
        for i in range(N):
            if i not in self.info_indices:
                temp[i] = np.inf
        right_matrix[:, 0] = temp

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for i in range(n):
                left_matrix[:, n - i - 1] = scf.bp_update_left(
                    left_matrix[:, n - i], right_matrix[:, n - i - 1], n - i
                )
            for i in range(n):
                right_matrix[:, i + 1] = scf.bp_update_right(
                    left_matrix[:, i + 1], right_matrix[:, i], i + 1
                )

            u_llr = left_matrix[:, 0] + right_matrix[:, 0]
            u_hat = (u_llr < 0).astype(int)
            u_hat[list(set(range(N)) - set(self.info_indices))] = 0

            x_llr = left_matrix[:, n] + right_matrix[:, n]
            x_hard = (x_llr < 0).astype(int)
            x_enc = polar_encode(u_hat)
            if np.array_equal(x_enc, x_hard):
                num_iters = it
                break

        u_llr = left_matrix[:, 0] + right_matrix[:, 0]
        u_hat = (u_llr < 0).astype(int)
        frozen_set = set(int(i) for i in range(N)) - set(int(i) for i in self.info_indices)
        for idx in frozen_set:
            u_hat[idx] = 0
        return u_hat, num_iters
