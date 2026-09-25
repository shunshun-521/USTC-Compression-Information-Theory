"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode


def _boxplus(x, y, alpha=0.9375, use_minsum=True):
    """Check-node update (box-plus)"""
    if use_minsum:
        return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))
    x = np.clip(x, -30.0, 30.0)
    y = np.clip(y, -30.0, 30.0)
    return np.log1p(np.exp(x + y)) - np.log(np.exp(x) + np.exp(y))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.llr_max = 30.0

        ind_range = np.arange(self.N // 2)
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

    def _single_iteration(self, ind_it, llr_ch, msg_l_prev, msg_r_in):
        msg_l_iter = [None] * (self.n + 1)
        msg_r_iter = [None] * (self.n + 1)
        zeros_half = np.zeros(self.N // 2)

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

            r1_out = _boxplus(r1_in, l2_in + r2_in, self.alpha, use_minsum=True)
            r2_out = _boxplus(r1_in, l1_in, self.alpha, use_minsum=True) + r2_in
            r_cat = np.concatenate([r1_out, r2_out])
            msg_r_iter[ind_s + 1] = r_cat[ind_inv]

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

            l1_out = _boxplus(l1_in, l2_in + r2_in, self.alpha, use_minsum=True)
            l2_out = _boxplus(r1_in, l1_in, self.alpha, use_minsum=True) + l2_in
            l_cat = np.concatenate([l1_out, l2_out])
            msg_l_iter[ind_s] = l_cat[ind_inv]

        return msg_l_iter, msg_r_iter

    def _hard_decision(self, msg_l):
        # 与 SC 一致：LLR >= 0 判为 0
        u_hat = np.zeros(self.N, dtype=int)
        u_hat[msg_l[0] < 0] = 1
        u_hat[self.frozen_idx] = 0
        return u_hat

    def _check_early_stop(self, llr_ch, msg_l):
        u_hat = self._hard_decision(msg_l)
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, hard_ch)

    def decode(self, llr_ch):
        llr_ch = np.clip(np.asarray(llr_ch, dtype=np.float64), -self.llr_max, self.llr_max)

        msg_r_in = np.zeros(self.N)
        msg_r_in[self.frozen_idx] = self.llr_max

        msg_l_prev = None
        msg_l_final = None
        num_iters = 0

        for ind_it in range(self.max_iter):
            msg_l_iter, _ = self._single_iteration(ind_it, llr_ch, msg_l_prev, msg_r_in)
            msg_l_prev = msg_l_iter
            msg_l_final = msg_l_iter
            num_iters = ind_it + 1
            if self._check_early_stop(llr_ch, msg_l_iter):
                break

        return self._hard_decision(msg_l_final), num_iters
