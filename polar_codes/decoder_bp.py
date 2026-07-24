"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode, bit_reversal_permutation
from channel import hard_decision_llr


def _initialize_connections(code_len):
    """构建因子图连接矩阵（参考 polar-ensembles）"""
    stages = int(math.log2(code_len))
    j0_mat = np.zeros((stages, code_len // 2))
    j1_mat = np.zeros((stages, code_len // 2))

    j0_mat[0, :] = np.arange(code_len // 2)
    j1_mat[0, :] = np.arange(code_len // 2) + code_len // 2

    for i in range(stages - 1):
        j0_mat[i + 1, :] = np.reshape(
            np.stack((j0_mat[i, : code_len // 4], j1_mat[i, : code_len // 4]), axis=1),
            (1, code_len // 2),
        )
        j1_mat[i + 1, :] = np.reshape(
            np.stack((j0_mat[i, code_len // 4 :], j1_mat[i, code_len // 4 :]), axis=1),
            (1, code_len // 2),
        )

    connections = np.ones((stages, code_len))
    for i in range(stages):
        connections[i, j1_mat[i, :].astype(int)] = 0
    return np.flipud(connections)


def _get_masks(code_len, connections):
    """获取上下分支掩码"""
    stages = int(math.log2(code_len))
    mask_dict = {}
    neg_mask_dict = {}
    for i in range(stages):
        mask = connections[i, :].astype(bool)
        mask_dict[i] = np.where(mask)[0]
        neg_mask_dict[i] = np.where(~mask)[0]
    return mask_dict, neg_mask_dict


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]
        connections = _initialize_connections(N)
        self.mask_dict, self.neg_mask_dict = _get_masks(N, connections)

    def _f_min_sum(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        brp = bit_reversal_permutation(self.N)
        llr_ch = llr_ch[brp]

        n = self.n
        N = self.N

        right = np.zeros((n + 1, N), dtype=np.float64)
        left = np.zeros((n + 1, N), dtype=np.float64)

        right[0, self.info_idx] = 0.0
        right[0, self.frozen_idx] = self.LARGE
        left[n, :] = llr_ch

        num_iters = 0
        for it in range(self.max_iter):
            num_iters = it + 1

            for stage in range(n):
                mask = self.mask_dict[stage]
                neg_mask = self.neg_mask_dict[stage]

                left_prev0 = left[stage + 1, neg_mask]
                left_prev1 = left[stage + 1, mask]
                right_prev0 = right[stage, neg_mask]
                right_prev1 = right[stage, mask]

                right[stage + 1, mask] = self._f_min_sum(
                    right_prev1, left_prev0 + right_prev0
                )
                right[stage + 1, neg_mask] = (
                    self._f_min_sum(right_prev1, left_prev1) + right_prev0
                )

            for stage in range(n - 1, -1, -1):
                mask = self.mask_dict[stage]
                neg_mask = self.neg_mask_dict[stage]

                left_prev0 = left[stage + 1, neg_mask]
                left_prev1 = left[stage + 1, mask]
                right_prev0 = right[stage, neg_mask]
                right_prev1 = right[stage, mask]

                left[stage, mask] = self._f_min_sum(
                    left_prev1, left_prev0 + right_prev0
                )
                left[stage, neg_mask] = (
                    self._f_min_sum(left_prev1, right_prev1) + left_prev0
                )

            right[0, self.info_idx] = 0.0
            right[0, self.frozen_idx] = self.LARGE

            total = left[0] + right[0]
            u_hat = (total < 0).astype(np.int32)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                break

        total = left[0] + right[0]
        u_hat = (total < 0).astype(np.int32)
        u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
