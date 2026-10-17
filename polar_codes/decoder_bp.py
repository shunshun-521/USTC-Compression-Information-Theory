"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（按极化码分层蝶形因子图实现）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.llr_max = 19.3
        self._build_stage_indices()

    def _build_stage_indices(self):
        ind_range = np.arange(self.N // 2)
        self.stage_ind_1 = []
        self.stage_ind_2 = []
        self.stage_ind_inv = []
        for ind_s in range(self.n):
            ind_1 = ind_range * 2 - np.mod(ind_range, 2**ind_s)
            ind_2 = ind_1 + 2**ind_s
            ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
            self.stage_ind_1.append(ind_1.astype(np.int64))
            self.stage_ind_2.append(ind_2.astype(np.int64))
            self.stage_ind_inv.append(ind_inv.astype(np.int64))

    def _boxplus(self, x, y):
        x = np.clip(x, -self.llr_max, self.llr_max)
        y = np.clip(y, -self.llr_max, self.llr_max)
        if self.alpha < 1.0:
            return self.alpha * f_operation(x, y)
        # 精确 box-plus（log-domain）
        return np.log1p(np.exp(x + y)) - np.log(np.exp(x) + np.exp(y))

    def _bp_iteration(self, llr_ch, msg_l_prev, msg_r_in):
        msg_l = [None] * (self.n + 1)
        msg_r = [None] * (self.n + 1)
        zeros_half = np.zeros(self.N // 2, dtype=np.float64)

        for ind_s in range(self.n):
            ind_1 = self.stage_ind_1[ind_s]
            ind_2 = self.stage_ind_2[ind_s]
            ind_inv = self.stage_ind_inv[ind_s]

            if ind_s == self.n - 1:
                l1_in = llr_ch[ind_1]
                l2_in = llr_ch[ind_2]
            elif msg_l_prev is None:
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
                r_in = msg_r[ind_s]
                r1_in = r_in[ind_1]
                r2_in = r_in[ind_2]

            r1_out = self._boxplus(r1_in, l2_in + r2_in)
            r2_out = self._boxplus(r1_in, l1_in) + r2_in
            r_out = np.concatenate([r1_out, r2_out])[ind_inv]
            msg_r[ind_s + 1] = r_out

        for ind_s in range(self.n - 1, -1, -1):
            ind_1 = self.stage_ind_1[ind_s]
            ind_2 = self.stage_ind_2[ind_s]
            ind_inv = self.stage_ind_inv[ind_s]

            if ind_s == self.n - 1:
                l1_in = llr_ch[ind_1]
                l2_in = llr_ch[ind_2]
            else:
                l_in = msg_l[ind_s + 1]
                l1_in = l_in[ind_1]
                l2_in = l_in[ind_2]

            if ind_s == 0:
                r1_in = msg_r_in[ind_1]
                r2_in = msg_r_in[ind_2]
            else:
                r_in = msg_r[ind_s]
                r1_in = r_in[ind_1]
                r2_in = r_in[ind_2]

            l1_out = self._boxplus(l1_in, l2_in + r2_in)
            l2_out = self._boxplus(r1_in, l1_in) + l2_in
            l_out = np.concatenate([l1_out, l2_out])[ind_inv]
            msg_l[ind_s] = l_out

        return msg_l, msg_r

    def decode(self, llr_ch):
        llr_ch = llr_ch.astype(np.float64)
        msg_r_in = np.zeros(self.N, dtype=np.float64)
        msg_r_in[self.frozen_bits] = self.llr_max

        msg_l_prev = None
        num_iters = self.max_iter
        u_hat = np.zeros(self.N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            msg_l, _ = self._bp_iteration(llr_ch, msg_l_prev, msg_r_in)
            msg_l_prev = msg_l

            u_llr = msg_l[0]
            u_hat = (u_llr < 0).astype(np.int8)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        return u_hat, num_iters
