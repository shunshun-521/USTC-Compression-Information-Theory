"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
（消息传递结构参考 Sionna PolarBPDecoder）
"""
import numpy as np
from encoder import polar_encode
from channel import hard_decision_llr
from decoder_sc import _sign


class BPDecoder:
    """BP 译码器（因子图 min-sum，含早停）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self._llr_max = 19.3

        ind_range = np.arange(N // 2)
        self._stage_ind_1 = []
        self._stage_ind_2 = []
        self._stage_ind_inv = []
        for ind_s in range(self.n):
            ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
            ind_2 = ind_1 + 2 ** ind_s
            ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
            self._stage_ind_1.append(ind_1)
            self._stage_ind_2.append(ind_2)
            self._stage_ind_inv.append(ind_inv)

    def _boxplus_minsum(self, x, y):
        x = np.clip(x, -self._llr_max, self._llr_max)
        y = np.clip(y, -self._llr_max, self._llr_max)
        return self.alpha * _sign(x) * _sign(y) * np.minimum(np.abs(x), np.abs(y))

    def _bp_single_iteration(self, ind_it, llr_ch, msg_l_prev, msg_r_in, zeros_half):
        msg_l_iter = [None] * (self.n + 1)
        msg_r_iter = [None] * (self.n + 1)

        for ind_s in range(self.n):
            ind_1 = self._stage_ind_1[ind_s]
            ind_2 = self._stage_ind_2[ind_s]
            ind_inv = self._stage_ind_inv[ind_s]

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

            r1_out = self._boxplus_minsum(r1_in, l2_in + r2_in)
            r2_out = self._boxplus_minsum(r1_in, l1_in) + r2_in

            r_out = np.concatenate([r1_out, r2_out])[ind_inv]
            msg_r_iter[ind_s + 1] = r_out

        for ind_s in range(self.n - 1, -1, -1):
            ind_1 = self._stage_ind_1[ind_s]
            ind_2 = self._stage_ind_2[ind_s]
            ind_inv = self._stage_ind_inv[ind_s]

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

            l1_out = self._boxplus_minsum(l1_in, l2_in + r2_in)
            l2_out = self._boxplus_minsum(r1_in, l1_in) + l2_in

            l_out = np.concatenate([l1_out, l2_out])[ind_inv]
            msg_l_iter[ind_s] = l_out

        return msg_l_iter, msg_r_iter

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        msg_r_in = np.zeros(self.N, dtype=np.float64)
        msg_r_in[self.frozen_idx] = -self._llr_max
        zeros_half = np.zeros(self.N // 2, dtype=np.float64)

        msg_l_prev = None
        num_iters = 0
        u_hat = np.zeros(self.N, dtype=int)

        for ind_it in range(self.max_iter):
            num_iters = ind_it + 1
            msg_l_iter, _ = self._bp_single_iteration(
                ind_it, llr_ch, msg_l_prev, msg_r_in, zeros_half
            )
            msg_l_prev = msg_l_iter

            soft = msg_l_iter[0]
            u_hat = (soft > 0).astype(int)  # Sionna hard_out 约定
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_decision_llr(llr_ch)):
                break

        return u_hat, num_iters
