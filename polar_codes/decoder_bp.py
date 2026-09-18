"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode


def _minsum_boxplus(x, y, alpha):
    x = np.clip(x, -1e10, 1e10)
    y = np.clip(y, -1e10, 1e10)
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器（Sionna 因子图布局 + min-sum）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n_stages = int(np.log2(N))
        frozen_bits = np.asarray(frozen_bits)
        self.frozen_bits = frozen_bits.astype(bool) if frozen_bits.dtype != bool else frozen_bits
        self.frozen_pos = np.where(self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.llr_max = 19.3
        self._build_stage_indices()

    def _build_stage_indices(self):
        ind_range = np.arange(self.N // 2)
        self.stage_ind_1 = []
        self.stage_ind_2 = []
        self.stage_ind_inv = []
        for ind_s in range(self.n_stages):
            ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
            ind_2 = ind_1 + 2 ** ind_s
            ind_inv = np.argsort(np.concatenate([ind_1, ind_2], axis=0))
            self.stage_ind_1.append(ind_1)
            self.stage_ind_2.append(ind_2)
            self.stage_ind_inv.append(ind_inv)

    def _bp_single_iteration(self, ind_it, llr_ch, msg_l_prev, msg_r_in):
        msg_l_iter = [None] * (self.n_stages + 1)
        msg_r_iter = [None] * (self.n_stages + 1)
        zeros_half = np.zeros(self.N // 2, dtype=np.float64)

        for ind_s in range(self.n_stages):
            ind_1 = self.stage_ind_1[ind_s]
            ind_2 = self.stage_ind_2[ind_s]
            ind_inv = self.stage_ind_inv[ind_s]

            if ind_s == self.n_stages - 1:
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

            r1_out = _minsum_boxplus(r1_in, l2_in + r2_in, self.alpha)
            r2_out = _minsum_boxplus(r1_in, l1_in, self.alpha) + r2_in
            r_out = np.concatenate([r1_out, r2_out])[ind_inv]
            msg_r_iter[ind_s + 1] = r_out

        for ind_s in range(self.n_stages - 1, -1, -1):
            ind_1 = self.stage_ind_1[ind_s]
            ind_2 = self.stage_ind_2[ind_s]
            ind_inv = self.stage_ind_inv[ind_s]

            if ind_s == self.n_stages - 1:
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

            l1_out = _minsum_boxplus(l1_in, l2_in + r2_in, self.alpha)
            l2_out = _minsum_boxplus(r1_in, l1_in, self.alpha) + l2_in
            l_out = np.concatenate([l1_out, l2_out])[ind_inv]
            msg_l_iter[ind_s] = l_out

        return msg_l_iter, msg_r_iter

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        msg_r_in = np.zeros(self.N, dtype=np.float64)
        msg_r_in[self.frozen_pos] = self.llr_max

        msg_l_prev = None
        num_iters = self.max_iter
        u_hat = np.zeros(self.N, dtype=int)

        for ind_it in range(self.max_iter):
            msg_l_iter, _ = self._bp_single_iteration(ind_it, llr_ch, msg_l_prev, msg_r_in)
            msg_l_prev = msg_l_iter

            total = msg_l_iter[0]
            u_hat = np.where(total >= 0.0, 0, 1).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_x = np.where(llr_ch >= 0.0, 0, 1).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = ind_it + 1
                break

        total = msg_l_prev[0]
        u_hat = np.where(total >= 0.0, 0, 1).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
