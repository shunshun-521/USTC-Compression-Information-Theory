"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _boxplus(a, b, alpha):
    """min-sum 校验节点更新"""
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_natural = np.where(frozen_bits.astype(bool))[0]
        self.info_natural = np.where(~frozen_bits.astype(bool))[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e7

    def _hard_decision(self, llr_left):
        u_hat = (llr_left < 0).astype(int)
        u_hat[self.frozen_natural] = 0
        return u_hat

    def decode(self, llr_ch):
        N = self.N
        n_stages = self.n
        alpha = self.alpha
        llr_ch = llr_ch.astype(np.float64)

        msg_l = [None] * (n_stages + 1)
        msg_r = [None] * (n_stages + 1)
        msg_r_in = np.zeros(N, dtype=np.float64)
        msg_r_in[self.frozen_natural] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for ind_it in range(self.max_iter):
            num_iters = ind_it + 1

            for ind_s in range(n_stages):
                ind_range = np.arange(N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
                ind_2 = ind_1 + 2 ** ind_s

                if ind_s == n_stages - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                elif ind_it == 0:
                    l1_in = np.zeros(N // 2, dtype=np.float64)
                    l2_in = np.zeros(N // 2, dtype=np.float64)
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

                r1_out = _boxplus(r1_in, l2_in + r2_in, alpha)
                r2_out = _boxplus(r1_in, l1_in, alpha) + r2_in

                r_out = np.zeros(N, dtype=np.float64)
                r_out[ind_1] = r1_out
                r_out[ind_2] = r2_out
                msg_r[ind_s + 1] = r_out

            for ind_s in range(n_stages - 1, -1, -1):
                ind_range = np.arange(N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
                ind_2 = ind_1 + 2 ** ind_s

                if ind_s == n_stages - 1:
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

                l1_out = _boxplus(l1_in, l2_in + r2_in, alpha)
                l2_out = _boxplus(r1_in, l1_in, alpha) + l2_in

                l_out = np.zeros(N, dtype=np.float64)
                l_out[ind_1] = l1_out
                l_out[ind_2] = l2_out
                msg_l[ind_s] = l_out

            u_hat = self._hard_decision(msg_l[0])
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        return u_hat, num_iters
