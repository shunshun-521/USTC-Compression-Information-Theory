"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math

from encoder import polar_encode
from decoder_sc import _prepare_llr


def _f_min_sum(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器（Sionna 因子图结构 + min-sum）。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n_stages = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_pos = np.where(self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha

        ind_range = np.arange(N // 2)
        self.stage_ind_1 = []
        self.stage_ind_2 = []
        self.stage_ind_inv = []
        for s in range(self.n_stages):
            ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** s)
            ind_2 = ind_1 + 2 ** s
            ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
            self.stage_ind_1.append(ind_1)
            self.stage_ind_2.append(ind_2)
            self.stage_ind_inv.append(ind_inv)

    def _single_iteration(self, llr_ch, msg_l_prev, msg_r_in, iteration_idx):
        n = self.n_stages
        N = self.N
        msg_l = [None] * (n + 1)
        msg_r = [None] * (n + 1)

        for s in range(n):
            ind_1 = self.stage_ind_1[s]
            ind_2 = self.stage_ind_2[s]
            ind_inv = self.stage_ind_inv[s]

            if s == n - 1:
                l1_in = llr_ch[ind_1]
                l2_in = llr_ch[ind_2]
            elif iteration_idx == 0:
                l1_in = np.zeros(len(ind_1))
                l2_in = np.zeros(len(ind_2))
            else:
                l_in = msg_l_prev[s + 1]
                l1_in = l_in[ind_1]
                l2_in = l_in[ind_2]

            if s == 0:
                r1_in = msg_r_in[ind_1]
                r2_in = msg_r_in[ind_2]
            else:
                r_in = msg_r[s]
                r1_in = r_in[ind_1]
                r2_in = r_in[ind_2]

            r1_out = _f_min_sum(r1_in, l2_in + r2_in, self.alpha)
            r2_out = _f_min_sum(r1_in, l1_in, self.alpha) + r2_in
            combined_idx = np.concatenate([ind_1, ind_2])
            r_out = np.zeros(N, dtype=np.float64)
            r_out[combined_idx] = np.concatenate([r1_out, r2_out])
            msg_r[s + 1] = r_out

        for s in range(n - 1, -1, -1):
            ind_1 = self.stage_ind_1[s]
            ind_2 = self.stage_ind_2[s]
            ind_inv = self.stage_ind_inv[s]

            if s == n - 1:
                l1_in = llr_ch[ind_1]
                l2_in = llr_ch[ind_2]
            else:
                l_in = msg_l[s + 1]
                l1_in = l_in[ind_1]
                l2_in = l_in[ind_2]

            if s == 0:
                r1_in = msg_r_in[ind_1]
                r2_in = msg_r_in[ind_2]
            else:
                r_in = msg_r[s]
                r1_in = r_in[ind_1]
                r2_in = r_in[ind_2]

            l1_out = _f_min_sum(l1_in, l2_in + r2_in, self.alpha)
            l2_out = _f_min_sum(r1_in, l1_in, self.alpha) + l2_in
            combined_idx = np.concatenate([ind_1, ind_2])
            l_out = np.zeros(N, dtype=np.float64)
            l_out[combined_idx] = np.concatenate([l1_out, l2_out])
            msg_l[s] = l_out

        return msg_l, msg_r

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_internal = _prepare_llr(llr_ch)
        N = self.N

        msg_r_in = np.zeros(N, dtype=np.float64)
        msg_r_in[self.frozen_pos] = self.LARGE

        msg_l_prev = [None] * (self.n_stages + 1)
        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(self.max_iter):
            msg_l, msg_r = self._single_iteration(llr_internal, msg_l_prev, msg_r_in, it)
            msg_l_prev = msg_l

            total = msg_l[0] + msg_r_in
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = it + 1
                break

        if msg_l_prev[0] is not None:
            total = msg_l_prev[0] + msg_r_in
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
