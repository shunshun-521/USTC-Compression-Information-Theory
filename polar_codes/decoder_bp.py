"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import f_operation
from encoder import bit_reversal_permutation, polar_encode


class BPDecoder:
    """BP 译码器（参考 Sionna/Arikan 因子图结构）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self._llr_max = 19.3

    def _boxplus_ms(self, x, y):
        x = np.clip(x, -self._llr_max, self._llr_max)
        y = np.clip(y, -self._llr_max, self._llr_max)
        return self.alpha * f_operation(x, y)

    def _stage_indices(self, stage):
        ind_range = np.arange(self.N // 2)
        ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** stage)
        ind_2 = ind_1 + 2 ** stage
        ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
        return ind_1, ind_2, ind_inv

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        llr = llr_ch[br].copy()

        msg_l = [None] * (self.n + 1)
        msg_r_hist = []
        msg_r_in = np.zeros(self.N, dtype=np.float64)
        msg_r_in[self.frozen_idx] = self._llr_max

        num_iters = self.max_iter
        u_hat = np.zeros(self.N, dtype=int)

        for it in range(self.max_iter):
            msg_r = [None] * (self.n + 1)

            for stage in range(self.n):
                ind_1, ind_2, ind_inv = self._stage_indices(stage)
                if stage == self.n - 1:
                    l1_in = llr[ind_1]
                    l2_in = llr[ind_2]
                elif it == 0:
                    l1_in = np.zeros(self.N // 2)
                    l2_in = np.zeros(self.N // 2)
                else:
                    l_prev = msg_l[stage + 1]
                    l1_in = l_prev[ind_1]
                    l2_in = l_prev[ind_2]

                if stage == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_prev = msg_r[stage]
                    r1_in = r_prev[ind_1]
                    r2_in = r_prev[ind_2]

                r1_out = self._boxplus_ms(r1_in, l2_in + r2_in)
                r2_out = self._boxplus_ms(r1_in, l1_in) + r2_in
                r_out = np.concatenate([r1_out, r2_out])[ind_inv]
                msg_r[stage + 1] = r_out

            for stage in range(self.n - 1, -1, -1):
                ind_1, ind_2, ind_inv = self._stage_indices(stage)
                if stage == self.n - 1:
                    l1_in = llr[ind_1]
                    l2_in = llr[ind_2]
                else:
                    l_prev = msg_l[stage + 1]
                    l1_in = l_prev[ind_1]
                    l2_in = l_prev[ind_2]

                if stage == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_prev = msg_r[stage]
                    r1_in = r_prev[ind_1]
                    r2_in = r_prev[ind_2]

                l1_out = self._boxplus_ms(l1_in, l2_in + r2_in)
                l2_out = self._boxplus_ms(r1_in, l1_in) + l2_in
                l_out = np.concatenate([l1_out, l2_out])[ind_inv]
                msg_l[stage] = l_out

            msg_r_hist.append(msg_r)

            llr_left = msg_l[0]
            u_soft = llr_left[self.info_idx]
            u_hat[self.info_idx] = (u_soft < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it + 1
                break

        return u_hat, num_iters
