"""
极化码 BP（置信传播）译码器
基于 Sionna / Arikan 因子图消息传递，min-sum 近似，含早停
"""
import math
import numpy as np

from encoder import polar_encode


def _boxplus_minsum(x, y, alpha):
    x = np.clip(x, -1e7, 1e7)
    y = np.clip(y, -1e7, 1e7)
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器（Sionna 风格索引与双程消息传递）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n_stages = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=np.int_)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.info_idx = np.where(self.frozen_bits == 0)[0]
        self.llr_max = 19.3

    def _stage_indices(self, ind_s):
        n = self.N
        ind_range = np.arange(n // 2)
        ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
        ind_2 = ind_1 + 2 ** ind_s
        ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
        return ind_1, ind_2, ind_inv

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.N
        n_stages = self.n_stages

        msg_r_in = np.zeros(n, dtype=np.float64)
        msg_r_in[self.frozen_idx] = self.llr_max

        msg_l = [None] * (n_stages + 1)
        msg_r = [None] * (n_stages + 1)

        num_iters = self.max_iter
        u_hat = np.zeros(n, dtype=np.int_)

        for ind_it in range(self.max_iter):
            for ind_s in range(n_stages):
                ind_1, ind_2, ind_inv = self._stage_indices(ind_s)

                if ind_s == n_stages - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                elif ind_it == 0:
                    l1_in = np.zeros(n // 2)
                    l2_in = np.zeros(n // 2)
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

                r1_out = _boxplus_minsum(r1_in, l2_in + r2_in, self.alpha)
                r2_out = _boxplus_minsum(r1_in, l1_in, self.alpha) + r2_in
                r_out = np.concatenate([r1_out, r2_out])[ind_inv]
                msg_r[ind_s + 1] = r_out

            for ind_s in range(n_stages - 1, -1, -1):
                ind_1, ind_2, ind_inv = self._stage_indices(ind_s)

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

                l1_out = _boxplus_minsum(l1_in, l2_in + r2_in, self.alpha)
                l2_out = _boxplus_minsum(r1_in, l1_in, self.alpha) + l2_in
                l_out = np.concatenate([l1_out, l2_out])[ind_inv]
                msg_l[ind_s] = l_out

            llr_u = msg_l[0]
            for i in self.info_idx:
                u_hat[i] = 0 if llr_u[i] >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(np.int_)
            if np.array_equal(x_hat, hard_ch):
                num_iters = ind_it + 1
                break

        llr_u = msg_l[0]
        u_hat = np.zeros(n, dtype=np.int_)
        u_hat[self.info_idx] = np.where(llr_u[self.info_idx] >= 0, 0, 1)
        return u_hat, num_iters
