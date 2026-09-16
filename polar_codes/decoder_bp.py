"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode


class BPDecoder:
    """BP 译码器（Sionna 风格 stage-wise 更新）"""

    LLR_MAX = 1e10

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_pos = np.where(self.frozen_bits == 1)[0]
        self.info_pos = np.where(self.frozen_bits == 0)[0]

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

    @staticmethod
    def _boxplus(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        abs_a = np.abs(a)
        abs_b = np.abs(b)
        sign = np.sign(a) * np.sign(b)
        return sign * np.minimum(abs_a, abs_b) + np.log1p(np.exp(-np.abs(a - b))) - np.log1p(
            np.exp(-(abs_a + abs_b))
        )

    def _f(self, a, b):
        return self.alpha * self._boxplus(a, b)

    def _single_iteration(self, ind_it, llr_ch, msg_l_prev, msg_r_in):
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
                r_in = msg_r[ind_s]
                r1_in = r_in[ind_1]
                r2_in = r_in[ind_2]

            r1_out = self._f(r1_in, l2_in + r2_in)
            r2_out = self._f(r1_in, l1_in) + r2_in
            r_out = np.empty(self.N, dtype=np.float64)
            r_out[np.concatenate([ind_1, ind_2])] = np.concatenate([r1_out, r2_out])
            msg_r[ind_s + 1] = r_out[ind_inv]

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

            l1_out = self._f(l1_in, l2_in + r2_in)
            l2_out = self._f(r1_in, l1_in) + l2_in
            l_out = np.empty(self.N, dtype=np.float64)
            l_out[np.concatenate([ind_1, ind_2])] = np.concatenate([l1_out, l2_out])
            msg_l[ind_s] = l_out[ind_inv]

        return msg_l, msg_r

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        msg_r_in = np.zeros(self.N, dtype=np.float64)
        msg_r_in[self.frozen_pos] = self.LLR_MAX

        msg_l_prev = None
        num_iters = self.max_iter

        for ind_it in range(self.max_iter):
            msg_l, _ = self._single_iteration(ind_it, llr_ch, msg_l_prev, msg_r_in)
            msg_l_prev = msg_l

            llr_left = msg_l[0]
            u_hat = np.zeros(self.N, dtype=int)
            u_hat[self.info_pos] = (llr_left[self.info_pos] < 0).astype(int)

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = ind_it + 1
                break

        llr_left = msg_l_prev[0]
        u_hat = np.zeros(self.N, dtype=int)
        u_hat[self.info_pos] = (llr_left[self.info_pos] < 0).astype(int)
        return u_hat, num_iters
