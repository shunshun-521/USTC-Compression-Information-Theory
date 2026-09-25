"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode


def _boxplus_min_sum(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器（stage-based 因子图实现）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]
        self.llr_max = 30.0

        ind_range = np.arange(N // 2)
        self.stage_ind_1 = []
        self.stage_ind_2 = []
        self.stage_ind_inv = []
        for s in range(self.n):
            ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** s)
            ind_2 = ind_1 + 2 ** s
            ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
            self.stage_ind_1.append(ind_1)
            self.stage_ind_2.append(ind_2)
            self.stage_ind_inv.append(ind_inv)

    def _bp_iteration(self, llr_ch, msg_l_prev, msg_r_in):
        msg_l = [None] * (self.n + 1)
        msg_r = [None] * (self.n + 1)
        zeros_half = np.zeros(self.N // 2)

        for s in range(self.n):
            ind_1 = self.stage_ind_1[s]
            ind_2 = self.stage_ind_2[s]
            ind_inv = self.stage_ind_inv[s]

            if s == self.n - 1:
                l1_in = llr_ch[ind_1]
                l2_in = llr_ch[ind_2]
            elif msg_l_prev is None:
                l1_in = zeros_half
                l2_in = zeros_half
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

            r1_out = _boxplus_min_sum(r1_in, l2_in + r2_in, self.alpha)
            r2_out = _boxplus_min_sum(r1_in, l1_in, self.alpha) + r2_in
            r_out = np.concatenate([r1_out, r2_out])[ind_inv]
            msg_r[s + 1] = r_out

        for s in range(self.n - 1, -1, -1):
            ind_1 = self.stage_ind_1[s]
            ind_2 = self.stage_ind_2[s]
            ind_inv = self.stage_ind_inv[s]

            if s == self.n - 1:
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

            l1_out = _boxplus_min_sum(l1_in, l2_in + r2_in, self.alpha)
            l2_out = _boxplus_min_sum(r1_in, l1_in, self.alpha) + l2_in
            l_out = np.concatenate([l1_out, l2_out])[ind_inv]
            msg_l[s] = l_out

        return msg_l, msg_r

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        llr_ch = -llr_ch  # 与 SC 译码器 LLR 约定对齐

        msg_r_in = np.zeros(N)
        msg_r_in[self.frozen_idx] = self.llr_max

        msg_l_prev = None
        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(self.max_iter):
            msg_l, msg_r = self._bp_iteration(llr_ch, msg_l_prev, msg_r_in)
            msg_l_prev = msg_l
            num_iters = it + 1

            soft = msg_l[0]
            u_hat = (soft <= 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch > 0).astype(int)  # llr negated: x=0 -> llr>0
            if np.array_equal(x_hat, hard_ch):
                break

        soft = msg_l_prev[0]
        u_hat = (soft <= 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
