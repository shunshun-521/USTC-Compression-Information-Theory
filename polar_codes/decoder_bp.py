"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import f_operation
from encoder import bit_reversal_permutation, polar_encode


class BPDecoder:
    """BP 译码器（参考 Sionna PolarBPDecoder 结构）"""

    LARGE = 1e7

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def _boxplus(self, a, b):
        return self.alpha * f_operation(a, b)

    def _stage_indices(self, stage):
        ind_range = np.arange(self.N // 2)
        ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** stage)
        ind_2 = ind_1 + 2 ** stage
        ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
        return ind_1, ind_2, ind_inv

    def decode(self, llr_ch):
        llr_nat = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        llr = llr_nat[br]

        N = self.N
        n = self.n

        msg_r_in = np.zeros(N, dtype=np.float64)
        msg_r_in[self.frozen] = self.LARGE

        msg_l = [None] * (n + 1)
        msg_r = [None] * (n + 1)

        num_iters = self.max_iter
        for it in range(self.max_iter):
            for stage in range(n):
                ind_1, ind_2, ind_inv = self._stage_indices(stage)

                if stage == n - 1:
                    l1_in = llr[ind_1]
                    l2_in = llr[ind_2]
                elif it == 0:
                    l1_in = np.zeros(N // 2)
                    l2_in = np.zeros(N // 2)
                else:
                    l1_in = msg_l[stage + 1][ind_1]
                    l2_in = msg_l[stage + 1][ind_2]

                if stage == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r1_in = msg_r[stage][ind_1]
                    r2_in = msg_r[stage][ind_2]

                r1_out = self._boxplus(r1_in, l2_in + r2_in)
                r2_out = self._boxplus(r1_in, l1_in) + r2_in
                msg_r[stage + 1] = np.concatenate([r1_out, r2_out])[ind_inv]

            for stage in range(n - 1, -1, -1):
                ind_1, ind_2, ind_inv = self._stage_indices(stage)

                if stage == n - 1:
                    l1_in = llr[ind_1]
                    l2_in = llr[ind_2]
                else:
                    l1_in = msg_l[stage + 1][ind_1]
                    l2_in = msg_l[stage + 1][ind_2]

                if stage == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r1_in = msg_r[stage][ind_1]
                    r2_in = msg_r[stage][ind_2]

                l1_out = self._boxplus(l1_in, l2_in + r2_in)
                l2_out = self._boxplus(r1_in, l1_in) + l2_in
                msg_l[stage] = np.concatenate([l1_out, l2_out])[ind_inv]

            total = msg_l[0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[total < 0] = 1
            u_hat[self.frozen] = 0

            x_hat = polar_encode(u_hat)
            hard = (llr_nat < 0).astype(int)
            if np.array_equal(x_hat, hard):
                num_iters = it + 1
                break
        else:
            total = msg_l[0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[total < 0] = 1
            u_hat[self.frozen] = 0

        return u_hat, num_iters
