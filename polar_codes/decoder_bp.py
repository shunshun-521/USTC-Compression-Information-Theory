"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _reorder_llr(llr_ch):
    br = bit_reversal_permutation(len(llr_ch))
    return np.asarray(llr_ch, dtype=np.float64)[br]


def _boxplus_minsum(x, y, alpha):
    x = np.clip(x, -30.0, 30.0)
    y = np.clip(y, -30.0, 30.0)
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器（参考 Sionna / Arikan BP 因子图结构）"""

    LLR_MAX = 19.3

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_pos = np.where(self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        llr_ch = _reorder_llr(llr_ch)
        n = self.n
        N = self.N
        num_iter = 0
        u_hat = np.zeros(N, dtype=int)

        msg_r_in = np.zeros(N, dtype=np.float64)
        msg_r_in[self.frozen_pos] = self.LLR_MAX

        msg_l = [None] * (n + 1)
        msg_r = [None] * (n + 1)

        for it in range(1, self.max_iter + 1):
            for stage in range(n):
                ind_range = np.arange(N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** stage)
                ind_2 = ind_1 + 2 ** stage
                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))

                if stage == n - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                elif it == 1:
                    l1_in = np.zeros(N // 2)
                    l2_in = np.zeros(N // 2)
                else:
                    l1_in = msg_l[stage + 1][ind_1]
                    l2_in = msg_l[stage + 1][ind_2]

                if stage == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r1_in = msg_r[stage][ind_1]
                    r2_in = msg_r[stage][ind_2]

                r1_out = _boxplus_minsum(r1_in, l2_in + r2_in, self.alpha)
                r2_out = _boxplus_minsum(r1_in, l1_in, self.alpha) + r2_in
                msg_r[stage + 1] = np.concatenate([r1_out, r2_out])[ind_inv]

            for stage in range(n - 1, -1, -1):
                ind_range = np.arange(N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** stage)
                ind_2 = ind_1 + 2 ** stage
                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))

                if stage == n - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                else:
                    l1_in = msg_l[stage + 1][ind_1]
                    l2_in = msg_l[stage + 1][ind_2]

                if stage == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r1_in = msg_r[stage][ind_1]
                    r2_in = msg_r[stage][ind_2]

                l1_out = _boxplus_minsum(l1_in, l2_in + r2_in, self.alpha)
                l2_out = _boxplus_minsum(r1_in, l1_in, self.alpha) + l2_in
                msg_l[stage] = np.concatenate([l1_out, l2_out])[ind_inv]

            llr_total = msg_l[0] + msg_r_in
            for i in range(N):
                u_hat[i] = 0 if llr_total[i] >= 0 else 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_x = (np.asarray(llr_ch, dtype=np.float64) < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break
            num_iters = it

        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
