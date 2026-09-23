"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation


class BPDecoder:
    """
    BP 译码器（参考 Sionna PolarBPDecoder 的因子图结构）。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]
        self._llr_max = 19.3

        ind_range = np.arange(N // 2)
        self._stage_ind_1 = []
        self._stage_ind_2 = []
        self._stage_ind_inv = []
        for ind_s in range(self.n):
            ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
            ind_2 = ind_1 + 2 ** ind_s
            ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
            self._stage_ind_1.append(ind_1)
            self._stage_ind_2.append(ind_2)
            self._stage_ind_inv.append(ind_inv)

    def _boxplus(self, x, y):
        """精确 box-plus 运算（数值稳定版）。"""
        x = np.clip(x, -self._llr_max, self._llr_max)
        y = np.clip(y, -self._llr_max, self._llr_max)
        return np.log1p(np.exp(x + y)) - np.logaddexp(x, y)

    def _f_ms(self, x, y):
        """min-sum 近似（用于可选加速）。"""
        return self.alpha * f_operation(x, y)

    def _bp_single_iteration(self, ind_it, llr_ch, msg_l_prev, msg_r_in):
        msg_l_iter = [None] * (self.n + 1)
        msg_r_iter = [None] * (self.n + 1)
        zeros_half = np.zeros(self.N // 2)

        for ind_s in range(self.n):
            ind_1 = self._stage_ind_1[ind_s]
            ind_2 = self._stage_ind_2[ind_s]
            ind_inv = self._stage_ind_inv[ind_s]

            if ind_s == self.n - 1:
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

        for ind_s in range(self.n - 1, -1, -1):
            ind_1 = self._stage_ind_1[ind_s]
            ind_2 = self._stage_ind_2[ind_s]
            ind_inv = self._stage_ind_inv[ind_s]

            if ind_s == self.n - 1:
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
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        brp = bit_reversal_permutation(self.N)
        llr_ch = llr_ch[brp]
        msg_r_in = np.zeros(self.N, dtype=np.float64)
        msg_r_in[self.frozen_idx] = self._llr_max

        msg_l_prev = None
        num_iters = self.max_iter
        final_l = None

        for ind_it in range(self.max_iter):
            msg_l_iter, _ = self._bp_single_iteration(ind_it, llr_ch, msg_l_prev, msg_r_in)
            msg_l_prev = msg_l_iter
            final_l = msg_l_iter[0]

            u_hat = self._hard_decision(final_l)
            if self._early_stop(u_hat, llr_ch):
                num_iters = ind_it + 1
                break

        u_hat = self._hard_decision(final_l)
        return u_hat, num_iters

    def _hard_decision(self, llr_left):
        u_hat = np.zeros(self.N, dtype=int)
        u_hat[self.info_idx] = (llr_left[self.info_idx] < 0).astype(int)
        return u_hat

    def _early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, hard_ch)
