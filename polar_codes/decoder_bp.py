"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _boxplus_minsum(x, y, alpha=0.9375):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """
    BP 译码器（因子图消息传递，参考 Arikan BP 结构）。
    """

    LLR_MAX = 19.3

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def _stage_indices(self, stage):
        ind_range = np.arange(self.N // 2)
        ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** stage)
        ind_2 = ind_1 + 2 ** stage
        ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
        return ind_1, ind_2, ind_inv

    def _hard_llr(self, llr_ch):
        return (llr_ch < 0).astype(int)

    def _check_early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        return np.array_equal(x_hat, self._hard_llr(llr_ch))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        n_stages = n

        msg_r_in = np.zeros(self.N, dtype=np.float64)
        msg_r_in[self.frozen_idx] = self.LLR_MAX

        msg_l = [None] * (n_stages + 1)
        msg_r = [None] * (n_stages + 1)

        num_iters = 0
        for ind_it in range(self.max_iter):
            num_iters = ind_it + 1
            msg_r_it = [None] * (n_stages + 1)
            for ind_s in range(n_stages):
                ind_1, ind_2, ind_inv = self._stage_indices(ind_s)
                if ind_s == n_stages - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                elif ind_it == 0:
                    l1_in = np.zeros(self.N // 2)
                    l2_in = np.zeros(self.N // 2)
                else:
                    l_in = msg_l[ind_s + 1]
                    l1_in = l_in[ind_1]
                    l2_in = l_in[ind_2]

                if ind_s == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_in = msg_r_it[ind_s]
                    r1_in = r_in[ind_1]
                    r2_in = r_in[ind_2]

                r1_out = _boxplus_minsum(r1_in, l2_in + r2_in, self.alpha)
                r2_out = _boxplus_minsum(r1_in, l1_in, self.alpha) + r2_in
                r_out = np.concatenate([r1_out, r2_out])[ind_inv]
                msg_r_it[ind_s + 1] = r_out

            msg_l_it = [None] * (n_stages + 1)
            for ind_s in range(n_stages - 1, -1, -1):
                ind_1, ind_2, ind_inv = self._stage_indices(ind_s)
                if ind_s == n_stages - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                else:
                    l_in = msg_l_it[ind_s + 1]
                    l1_in = l_in[ind_1]
                    l2_in = l_in[ind_2]

                if ind_s == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_in = msg_r_it[ind_s]
                    r1_in = r_in[ind_1]
                    r2_in = r_in[ind_2]

                l1_out = _boxplus_minsum(l1_in, l2_in + r2_in, self.alpha)
                l2_out = _boxplus_minsum(r1_in, l1_in, self.alpha) + l2_in
                l_out = np.concatenate([l1_out, l2_out])[ind_inv]
                msg_l_it[ind_s] = l_out

            msg_l = msg_l_it
            msg_r = msg_r_it

            posterior = msg_l[0]
            u_hat = np.zeros(self.N, dtype=int)
            u_hat[posterior < 0] = 1
            u_hat[self.frozen_idx] = 0
            if self._check_early_stop(u_hat, llr_ch):
                break

        posterior = msg_l[0]
        u_hat = np.zeros(self.N, dtype=int)
        u_hat[posterior < 0] = 1
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
