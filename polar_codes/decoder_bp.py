"""
极化码 BP（置信传播）译码器
基于因子图（Sionna/Arikan 风格），使用 box-plus，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode


def _boxplus(x, y, llr_max=19.3):
    x = np.clip(x, -llr_max, llr_max)
    y = np.clip(y, -llr_max, llr_max)
    return np.log1p(np.exp(x + y)) - np.log(np.exp(x) + np.exp(y))


def _precompute_stage_indices(n_stages, n):
    ind_range = np.arange(n // 2)
    stage_ind_1 = []
    stage_ind_2 = []
    stage_ind_inv = []
    for ind_s in range(n_stages):
        ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
        ind_2 = ind_1 + 2 ** ind_s
        ind_inv = np.argsort(np.concatenate([ind_1, ind_2], axis=0))
        stage_ind_1.append(ind_1)
        stage_ind_2.append(ind_2)
        stage_ind_inv.append(ind_inv)
    return stage_ind_1, stage_ind_2, stage_ind_inv


class BPDecoder:
    """BP 译码器。"""

    LLR_MAX = 19.3

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha  # kept for API compatibility
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]
        self.stage_ind_1, self.stage_ind_2, self.stage_ind_inv = _precompute_stage_indices(
            self.n, N
        )

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

            r1_out = _boxplus(r1_in, l2_in + r2_in, self.LLR_MAX)
            r2_out = _boxplus(r1_in, l1_in, self.LLR_MAX) + r2_in
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

            l1_out = _boxplus(l1_in, l2_in + r2_in, self.LLR_MAX)
            l2_out = _boxplus(r1_in, l1_in, self.LLR_MAX) + l2_in
            l_out = np.concatenate([l1_out, l2_out])[ind_inv]
            msg_l_iter[ind_s] = l_out

        return msg_l_iter, msg_r_iter

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        msg_r_in = np.zeros(self.N, dtype=np.float64)
        msg_r_in[self.frozen_idx] = self.LLR_MAX

        msg_l_prev = None
        num_iters = 0
        u_hat = np.zeros(self.N, dtype=int)

        for ind_it in range(self.max_iter):
            msg_l_iter, _ = self._bp_single_iteration(ind_it, llr_ch, msg_l_prev, msg_r_in)
            msg_l_prev = msg_l_iter
            num_iters = ind_it + 1

            llr_left = msg_l_iter[0]
            for i in range(self.N):
                u_hat[i] = 0 if llr_left[i] > 0 else 1
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        llr_left = msg_l_prev[0]
        for i in range(self.N):
            u_hat[i] = 0 if llr_left[i] > 0 else 1
        u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
