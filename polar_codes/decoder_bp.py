"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from channel import hard_decision_llr
from decoder_sc import align_llr_for_decoder


def _precompute_stage_indices(n_stages):
    ind_range = np.arange(2 ** (n_stages - 1))
    stage_ind_1, stage_ind_2, stage_ind_inv = [], [], []
    for ind_s in range(n_stages):
        ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
        ind_2 = ind_1 + 2 ** ind_s
        ind_inv = np.argsort(np.concatenate([ind_1, ind_2], axis=0))
        stage_ind_1.append(ind_1)
        stage_ind_2.append(ind_2)
        stage_ind_inv.append(ind_inv)
    return stage_ind_1, stage_ind_2, stage_ind_inv


def _boxplus(x, y, llr_max=19.3):
    x = np.clip(x, -llr_max, llr_max)
    y = np.clip(y, -llr_max, llr_max)
    return np.log1p(np.exp(x + y)) - np.log(np.exp(x) + np.exp(y))


def _min_sum(a, b, alpha):
    sa = np.sign(a)
    sb = np.sign(b)
    sa = np.where(sa == 0, 1.0, sa)
    sb = np.where(sb == 0, 1.0, sb)
    return alpha * sa * sb * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（min-sum / boxplus 可切换，含早停）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375, use_boxplus=False):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_pos = np.where(self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.use_boxplus = use_boxplus
        self._llr_max = 19.3
        self._stage_ind_1, self._stage_ind_2, self._stage_ind_inv = _precompute_stage_indices(
            self.n
        )

    def _cn(self, a, b):
        if self.use_boxplus:
            return _boxplus(a, b, self._llr_max)
        return _min_sum(a, b, self.alpha)

    def _bp_iteration(self, ind_it, llr_ch, msg_l_prev, msg_r_in):
        n = self.n
        msg_l_iter = [None] * (n + 1)
        msg_r_iter = [None] * (n + 1)
        zeros_half = np.zeros(self.N // 2, dtype=np.float64)

        for ind_s in range(n):
            ind_1 = self._stage_ind_1[ind_s]
            ind_2 = self._stage_ind_2[ind_s]
            ind_inv = self._stage_ind_inv[ind_s]

            if ind_s == n - 1:
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

            r1_out = self._cn(r1_in, l2_in + r2_in)
            r2_out = self._cn(r1_in, l1_in) + r2_in
            r_out = np.concatenate([r1_out, r2_out])[ind_inv]
            msg_r_iter[ind_s + 1] = r_out

        for ind_s in range(n - 1, -1, -1):
            ind_1 = self._stage_ind_1[ind_s]
            ind_2 = self._stage_ind_2[ind_s]
            ind_inv = self._stage_ind_inv[ind_s]

            if ind_s == n - 1:
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

            l1_out = self._cn(l1_in, l2_in + r2_in)
            l2_out = self._cn(r1_in, l1_in) + l2_in
            l_out = np.concatenate([l1_out, l2_out])[ind_inv]
            msg_l_iter[ind_s] = l_out

        return msg_l_iter, msg_r_iter

    def decode(self, llr_ch):
        llr_nat = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = align_llr_for_decoder(llr_nat)
        msg_r_in = np.zeros(self.N, dtype=np.float64)
        msg_r_in[self.frozen_pos] = self._llr_max

        msg_l_prev = None
        num_iters = 0
        u_hat = np.zeros(self.N, dtype=np.int8)

        for ind_it in range(self.max_iter):
            num_iters = ind_it + 1
            msg_l_iter, _ = self._bp_iteration(ind_it, llr_ch, msg_l_prev, msg_r_in)
            msg_l_prev = msg_l_iter

            posterior = msg_l_iter[0]
            u_hat = np.where(posterior >= 0, 0, 1).astype(np.int8)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_decision_llr(llr_nat)):
                break

        posterior = msg_l_prev[0]
        u_hat = np.where(posterior >= 0, 0, 1).astype(np.int8)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
