"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from channel import hard_decision_llr
from decoder_sc import _permute_llr_for_decode, f_operation
from encoder import polar_encode


def bp_f_operation(La, Lb, alpha=0.9375):
    """min-sum f 运算，带缩放因子 alpha"""
    return alpha * f_operation(La, Lb)


class BPDecoder:
    """BP 译码器（Sionna 风格因子图索引）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]
        self.llr_max = 1e6

        ind_range = np.arange(N // 2)
        self.stage_ind_1 = []
        self.stage_ind_2 = []
        self.stage_ind_inv = []
        for ind_s in range(self.n):
            ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
            ind_2 = ind_1 + 2 ** ind_s
            ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
            self.stage_ind_1.append(ind_1)
            self.stage_ind_2.append(ind_2)
            self.stage_ind_inv.append(ind_inv)

    def _boxplus(self, x, y):
        x = np.clip(x, -self.llr_max, self.llr_max)
        y = np.clip(y, -self.llr_max, self.llr_max)
        return bp_f_operation(x, y, self.alpha)

    def _bp_single_iteration(self, ind_it, llr_ch, msg_l_prev, msg_r_in):
        n_stages = self.n
        msg_l_iter = [None] * (n_stages + 1)
        msg_r_iter = [None] * (n_stages + 1)
        zeros_half = np.zeros(self.N // 2)

        for ind_s in range(n_stages):
            ind_1 = self.stage_ind_1[ind_s]
            ind_2 = self.stage_ind_2[ind_s]
            ind_inv = self.stage_ind_inv[ind_s]

            if ind_s == n_stages - 1:
                l1_in = llr_ch[ind_1]
                l2_in = llr_ch[ind_2]
            elif ind_it == 0:
                l1_in = zeros_half
                l2_in = zeros_half
            else:
                l_in = msg_l_prev[ind_s + 1]
                l1_in = l_in[ind_1]
                l2_in = l_in[ind_2]

            if ind_s == 0:
                r1_in = msg_r_in[ind_1]
                r2_in = msg_r_in[ind_2]
            else:
                r_in = msg_r_iter[ind_s]
                r1_in = r_in[ind_1]
                r2_in = r_in[ind_2]

            r1_out = self._boxplus(r1_in, l2_in + r2_in)
            r2_out = self._boxplus(r1_in, l1_in) + r2_in
            r_out = np.concatenate([r1_out, r2_out])[ind_inv]
            msg_r_iter[ind_s + 1] = r_out

        for ind_s in range(n_stages - 1, -1, -1):
            ind_1 = self.stage_ind_1[ind_s]
            ind_2 = self.stage_ind_2[ind_s]
            ind_inv = self.stage_ind_inv[ind_s]

            if ind_s == n_stages - 1:
                l1_in = llr_ch[ind_1]
                l2_in = llr_ch[ind_2]
            else:
                l_in = msg_l_iter[ind_s + 1]
                l1_in = l_in[ind_1]
                l2_in = l_in[ind_2]

            if ind_s == 0:
                r1_in = msg_r_in[ind_1]
                r2_in = msg_r_in[ind_2]
            else:
                r_in = msg_r_iter[ind_s]
                r1_in = r_in[ind_1]
                r2_in = r_in[ind_2]

            l1_out = self._boxplus(l1_in, l2_in + r2_in)
            l2_out = self._boxplus(r1_in, l1_in) + l2_in
            l_out = np.concatenate([l1_out, l2_out])[ind_inv]
            msg_l_iter[ind_s] = l_out

        return msg_l_iter, msg_r_iter

    def decode(self, llr_ch):
        llr_orig = np.asarray(llr_ch, dtype=np.float64)
        llr_dec = _permute_llr_for_decode(llr_orig)
        N = self.N

        msg_r_in = np.zeros(N, dtype=np.float64)
        msg_r_in[self.frozen_idx] = self.llr_max

        msg_l_prev = None
        msg_l_last = None

        for ind_it in range(self.max_iter):
            msg_l_iter, _ = self._bp_single_iteration(ind_it, llr_dec, msg_l_prev, msg_r_in)
            msg_l_prev = msg_l_iter
            msg_l_last = msg_l_iter

            total_llr = msg_l_last[0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[self.info_idx] = (total_llr[self.info_idx] < 0).astype(int)

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_orig)
            if np.array_equal(x_hat, x_hard):
                return u_hat, ind_it + 1

        total_llr = msg_l_last[0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[self.info_idx] = (total_llr[self.info_idx] < 0).astype(int)
        return u_hat, self.max_iter
