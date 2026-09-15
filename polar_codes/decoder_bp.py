"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode, bit_reversal_permutation
from channel import hard_decision_llr


class BPDecoder:
    """BP 译码器（参考 Sionna 因子图结构，min-sum 近似）。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        frozen_bits = np.asarray(frozen_bits)
        if frozen_bits.dtype == bool:
            self.frozen_mask = frozen_bits.copy()
        else:
            self.frozen_mask = frozen_bits.astype(int) != 0
        self.frozen_indices = np.where(self.frozen_mask)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.llr_max = 19.3

        ind_range = np.arange(self.N // 2)
        self.stage_ind_1 = []
        self.stage_ind_2 = []
        self.stage_ind_inv = []
        for ind_s in range(self.n):
            ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
            ind_2 = ind_1 + 2 ** ind_s
            ind_inv = np.argsort(np.concatenate([ind_1, ind_2], axis=0))
            self.stage_ind_1.append(ind_1)
            self.stage_ind_2.append(ind_2)
            self.stage_ind_inv.append(ind_inv)
        self._rev = bit_reversal_permutation(N)

    def _boxplus_ms(self, x, y):
        x = np.clip(x, -self.llr_max, self.llr_max)
        y = np.clip(y, -self.llr_max, self.llr_max)
        return self.alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))

    def _bp_single_iteration(self, ind_it, llr_ch, msg_l_prev, msg_r_in):
        msg_l_iter = [None] * (self.n + 1)
        msg_r_iter = [None] * (self.n + 1)
        zeros_half = np.zeros(self.N // 2, dtype=np.float64)

        for ind_s in range(self.n):
            ind_1 = self.stage_ind_1[ind_s]
            ind_2 = self.stage_ind_2[ind_s]
            ind_inv = self.stage_ind_inv[ind_s]

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

            r1_out = self._boxplus_ms(r1_in, l2_in + r2_in)
            r2_out = self._boxplus_ms(r1_in, l1_in) + r2_in
            r_out = np.concatenate([r1_out, r2_out])[ind_inv]
            msg_r_iter[ind_s + 1] = r_out

        for ind_s in range(self.n - 1, -1, -1):
            ind_1 = self.stage_ind_1[ind_s]
            ind_2 = self.stage_ind_2[ind_s]
            ind_inv = self.stage_ind_inv[ind_s]

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

            l1_out = self._boxplus_ms(l1_in, l2_in + r2_in)
            l2_out = self._boxplus_ms(r1_in, l1_in) + l2_in
            l_out = np.concatenate([l1_out, l2_out])[ind_inv]
            msg_l_iter[ind_s] = l_out

        return msg_l_iter, msg_r_iter

    def _permute_llr(self, llr_ch):
        llr_perm = np.empty(self.N, dtype=np.float64)
        for k in range(self.N):
            llr_perm[k] = llr_ch[self._rev[k]]
        return llr_perm

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_internal = self._permute_llr(llr_ch)
        msg_r_in = np.zeros(self.N, dtype=np.float64)
        msg_r_in[self.frozen_indices] = self.LARGE

        msg_l_prev = None
        num_iters = self.max_iter
        u_hat = np.zeros(self.N, dtype=int)

        for ind_it in range(self.max_iter):
            msg_l_iter, _ = self._bp_single_iteration(ind_it, llr_internal, msg_l_prev, msg_r_in)
            msg_l_prev = msg_l_iter

            llr_total = msg_l_iter[0]
            u_hat = (llr_total < 0).astype(int)
            u_hat[self.frozen_mask] = 0

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_decision_llr(llr_ch)):
                num_iters = ind_it + 1
                break

        llr_total = msg_l_prev[0]
        u_hat = (llr_total < 0).astype(int)
        u_hat[self.frozen_mask] = 0
        return u_hat, num_iters
