"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import _align_channel_llrs, f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器（基于极化码因子图的分层消息传递）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_pos = np.where(self.frozen_bits)[0]
        self.info_pos = np.where(~self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.llr_max = 30.0

    def _boxplus(self, x, y):
        x = np.clip(x, -self.llr_max, self.llr_max)
        y = np.clip(y, -self.llr_max, self.llr_max)
        return self.alpha * f_operation(x, y)

    def _stage_indices(self, stage):
        ind_range = np.arange(self.N // 2)
        ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** stage)
        ind_2 = ind_1 + 2 ** stage
        ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
        return ind_1, ind_2, ind_inv

    def decode(self, llr_ch):
        llr_ch_orig = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = np.clip(_align_channel_llrs(llr_ch_orig), -self.llr_max, self.llr_max)

        msg_r_in = np.zeros(self.N, dtype=np.float64)
        msg_r_in[self.frozen_pos] = self.llr_max

        msg_l = [None] * (self.n + 1)
        msg_r = [None] * (self.n + 1)
        num_iters = self.max_iter

        for it in range(self.max_iter):
            for stage in range(self.n):
                ind_1, ind_2, ind_inv = self._stage_indices(stage)

                if stage == self.n - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                elif it == 0:
                    l1_in = np.zeros(self.N // 2)
                    l2_in = np.zeros(self.N // 2)
                else:
                    l_in = msg_l[stage + 1]
                    l1_in = l_in[ind_1]
                    l2_in = l_in[ind_2]

                if stage == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_in = msg_r[stage]
                    r1_in = r_in[ind_1]
                    r2_in = r_in[ind_2]

                r1_out = self._boxplus(r1_in, l2_in + r2_in)
                r2_out = self._boxplus(r1_in, l1_in) + r2_in
                msg_r[stage + 1] = np.concatenate([r1_out, r2_out])[ind_inv]

            for stage in range(self.n - 1, -1, -1):
                ind_1, ind_2, ind_inv = self._stage_indices(stage)

                if stage == self.n - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                else:
                    l_in = msg_l[stage + 1]
                    l1_in = l_in[ind_1]
                    l2_in = l_in[ind_2]

                if stage == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_in = msg_r[stage]
                    r1_in = r_in[ind_1]
                    r2_in = r_in[ind_2]

                l1_out = self._boxplus(l1_in, l2_in + r2_in)
                l2_out = self._boxplus(r1_in, l1_in) + l2_in
                msg_l[stage] = np.concatenate([l1_out, l2_out])[ind_inv]

            u_hat = np.zeros(self.N, dtype=int)
            u_hat[self.info_pos] = (msg_l[0][self.info_pos] < 0).astype(int)
            u_hat[self.frozen_pos] = 0

            if np.array_equal(polar_encode(u_hat), (llr_ch_orig < 0).astype(int)):
                num_iters = it + 1
                break

        u_hat = np.zeros(self.N, dtype=int)
        u_hat[self.info_pos] = (msg_l[0][self.info_pos] < 0).astype(int)
        u_hat[self.frozen_pos] = 0
        return u_hat, num_iters
