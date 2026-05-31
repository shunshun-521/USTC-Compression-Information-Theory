"""
极化码 BP（置信传播）译码器
基于因子图 min-sum，结构参考 Sionna PolarBPDecoder
"""
import numpy as np
from encoder import polar_encode


def _boxplus(x, y, alpha=0.9375, llr_max=30.0):
    x = np.clip(x, -llr_max, llr_max)
    y = np.clip(y, -llr_max, llr_max)
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n_stages = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_pos = np.where(self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.llr_max = 30.0

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.N
        alpha = self.alpha

        msg_r_in = np.zeros(n, dtype=np.float64)
        msg_r_in[self.frozen_pos] = self.llr_max

        msg_l_prev = [None] * (self.n_stages + 1)
        num_iters_done = self.max_iter

        for ind_it in range(self.max_iter):
            msg_l = [None] * (self.n_stages + 1)
            msg_r = [None] * (self.n_stages + 1)
            msg_r[0] = msg_r_in.copy()

            for ind_s in range(self.n_stages):
                ind_range = np.arange(n // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2**ind_s)
                ind_2 = ind_1 + 2**ind_s

                if ind_s == self.n_stages - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                elif ind_it == 0:
                    l1_in = np.zeros(n // 2)
                    l2_in = np.zeros(n // 2)
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

                r1_out = _boxplus(r1_in, l2_in + r2_in, alpha, self.llr_max)
                r2_out = _boxplus(r1_in, l1_in, alpha, self.llr_max) + r2_in
                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
                msg_r[ind_s + 1] = np.concatenate([r1_out, r2_out])[ind_inv]

            for ind_s in range(self.n_stages - 1, -1, -1):
                ind_range = np.arange(n // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2**ind_s)
                ind_2 = ind_1 + 2**ind_s

                if ind_s == self.n_stages - 1:
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

                l1_out = _boxplus(l1_in, l2_in + r2_in, alpha, self.llr_max)
                l2_out = _boxplus(l1_in, r1_in, alpha, self.llr_max) + l2_in
                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
                msg_l[ind_s] = np.concatenate([l1_out, l2_out])[ind_inv]

            msg_l_prev = msg_l

            total = msg_l[0] + msg_r[0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0
            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, (llr_ch < 0).astype(int)):
                num_iters_done = ind_it + 1
                break

        total = msg_l_prev[0] + msg_r[0]
        u_hat = np.zeros(n, dtype=int)
        u_hat[total < 0] = 1
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters_done
