"""
极化码 BP（置信传播）译码器
基于 Sionna/Arikan 因子图 flooding，min-sum 近似，含早停机制
"""
import math

import numpy as np

from channel import hard_decision_llr
from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器（Sionna 风格 flooding + min-sum）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.info_idx = np.where(self.frozen_bits == 0)[0]
        self.llr_max = 19.3

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
        llr_ch = np.clip(np.asarray(llr_ch, dtype=np.float64), -self.llr_max, self.llr_max)
        n = self.n
        N = self.N
        num_iter = self.max_iter

        msg_r_in = np.zeros(N, dtype=np.float64)
        msg_r_in[self.frozen_idx] = self.llr_max

        msg_l = [[None] * (n + 1) for _ in range(num_iter)]
        msg_r = [[None] * (n + 1) for _ in range(num_iter)]
        x_hard = hard_decision_llr(llr_ch)

        for ind_it in range(num_iter):
            # 左到右：更新 R 消息
            for ind_s in range(n):
                ind_1, ind_2, ind_inv = self._stage_indices(ind_s)

                if ind_s == n - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                elif ind_it == 0:
                    l1_in = np.zeros(N // 2)
                    l2_in = np.zeros(N // 2)
                else:
                    l_in = msg_l[ind_it - 1][ind_s + 1]
                    l1_in = l_in[ind_1]
                    l2_in = l_in[ind_2]

                if ind_s == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_in = msg_r[ind_it][ind_s]
                    r1_in = r_in[ind_1]
                    r2_in = r_in[ind_2]

                r1_out = self._boxplus(r1_in, l2_in + r2_in)
                r2_out = self._boxplus(r1_in, l1_in) + r2_in
                msg_r[ind_it][ind_s + 1] = np.concatenate([r1_out, r2_out])[ind_inv]

            # 右到左：更新 L 消息
            for ind_s in range(n - 1, -1, -1):
                ind_1, ind_2, ind_inv = self._stage_indices(ind_s)

                if ind_s == n - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                else:
                    l_in = msg_l[ind_it][ind_s + 1]
                    l1_in = l_in[ind_1]
                    l2_in = l_in[ind_2]

                if ind_s == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_in = msg_r[ind_it][ind_s]
                    r1_in = r_in[ind_1]
                    r2_in = r_in[ind_2]

                l1_out = self._boxplus(l1_in, l2_in + r2_in)
                l2_out = self._boxplus(r1_in, l1_in) + l2_in
                msg_l[ind_it][ind_s] = np.concatenate([l1_out, l2_out])[ind_inv]

            soft_full = msg_l[ind_it][0]
            u_hat = (soft_full < 0).astype(int)
            u_hat[self.frozen_idx] = 0
            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, x_hard):
                return u_hat, ind_it + 1

        soft_full = msg_l[num_iter - 1][0]
        u_hat = (soft_full < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iter
