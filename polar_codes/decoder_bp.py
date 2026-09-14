"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（参考 Sionna 因子图索引结构）。"""

    LLR_MAX = 19.3

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._precompute_stage_indices()

    def _precompute_stage_indices(self):
        ind_range = np.arange(self.N // 2)
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

    def _boxplus(self, a, b):
        return self.alpha * f_operation(a, b)

    def _hard_decision(self, msg_l):
        # 与 Sionna 一致：内部 LLR 正号表示比特 0
        u_hat = (msg_l[0] > 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat

    def _single_iteration(self, ind_it, llr_ch, msg_l_prev, msg_r_in):
        n = self.n
        msg_l = [None] * (n + 1)
        msg_r = [None] * (n + 1)
        zeros_half = np.zeros(self.N // 2)

        for s in range(n):
            ind_1 = self.stage_ind_1[s]
            ind_2 = self.stage_ind_2[s]
            ind_inv = self.stage_ind_inv[s]

            if s == n - 1:
                l1_in = llr_ch[ind_1]
                l2_in = llr_ch[ind_2]
            elif ind_it == 0:
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

            r1_out = self._boxplus(r1_in, l2_in + r2_in)
            r2_out = self._boxplus(r1_in, l1_in) + r2_in
            r_out = np.concatenate([r1_out, r2_out])[ind_inv]
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

            l1_out = self._boxplus(l1_in, l2_in + r2_in)
            l2_out = self._boxplus(r1_in, l1_in) + l2_in
            l_out = np.concatenate([l1_out, l2_out])[ind_inv]
            msg_l[s] = l_out

        return msg_l, msg_r

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)。"""
        llr_ch = -np.asarray(llr_ch, dtype=np.float64)
        msg_r_in = np.zeros(self.N, dtype=np.float64)
        msg_r_in[self.frozen_bits] = self.LLR_MAX

        msg_l_prev = None
        num_iters = 0
        for ind_it in range(self.max_iter):
            num_iters = ind_it + 1
            msg_l, msg_r = self._single_iteration(ind_it, llr_ch, msg_l_prev, msg_r_in)
            msg_l_prev = msg_l

            u_hat = self._hard_decision(msg_l)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        u_hat = self._hard_decision(msg_l)
        return u_hat, num_iters
