"""
极化码 BP（置信传播）译码器
基于 Sionna/Arikan 因子图，使用 box-plus，含早停机制
"""
import math

import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _boxplus(x, y, llr_max=19.3):
    x_in = np.clip(x, -llr_max, llr_max)
    y_in = np.clip(y, -llr_max, llr_max)
    return np.log(1.0 + np.exp(x_in + y_in)) - np.log(np.exp(x_in) + np.exp(y_in))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n_stages = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_pos = np.where(self.frozen_bits)[0]
        self.max_iter = max_iter
        self.llr_max = 19.3
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = llr_ch[self.br]
        N = self.N
        n_stages = self.n_stages
        num_iter = self.max_iter

        msg_r_in = np.zeros(N, dtype=np.float64)
        msg_r_in[self.frozen_pos] = self.llr_max

        msg_l = [[None] * (n_stages + 1) for _ in range(num_iter)]
        msg_r = [[None] * (n_stages + 1) for _ in range(num_iter)]

        for ind_it in range(num_iter):
            for ind_s in range(n_stages):
                ind_range = np.arange(N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
                ind_2 = ind_1 + 2 ** ind_s

                if ind_s == n_stages - 1:
                    l1_in = llr[ind_1]
                    l2_in = llr[ind_2]
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

                r1_out = _boxplus(r1_in, l2_in + r2_in, self.llr_max)
                r2_out = _boxplus(r1_in, l1_in, self.llr_max) + r2_in

                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
                r_out = np.concatenate([r1_out, r2_out])[ind_inv]
                msg_r[ind_it][ind_s + 1] = r_out

            for ind_s in range(n_stages - 1, -1, -1):
                ind_range = np.arange(N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
                ind_2 = ind_1 + 2 ** ind_s
                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))

                if ind_s == n_stages - 1:
                    l1_in = llr[ind_1]
                    l2_in = llr[ind_2]
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

                l1_out = _boxplus(l1_in, l2_in + r2_in, self.llr_max)
                l2_out = _boxplus(r1_in, l1_in, self.llr_max) + l2_in
                l_out = np.concatenate([l1_out, l2_out])[ind_inv]
                msg_l[ind_it][ind_s] = l_out

            total = msg_l[ind_it][0] + msg_r_in
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0
            if np.array_equal(polar_encode(u_hat), (llr_ch < 0).astype(int)):
                return u_hat, ind_it + 1

        total = msg_l[num_iter - 1][0] + msg_r_in
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iter
