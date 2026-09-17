"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _f_min_sum(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """
    BP 译码器。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_pos = np.where(self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.llr_max = 19.3
        self.br = bit_reversal_permutation(N)

    def _stage_indices(self, stage):
        ind_range = np.arange(self.N // 2)
        ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** stage)
        ind_2 = ind_1 + 2 ** stage
        ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
        return ind_1, ind_2, ind_inv

    def _hard_bits(self, msg_l, msg_r):
        total = msg_l[0] + msg_r[0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat

    def _check_early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, hard_ch)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_dec = np.clip(llr_ch[self.br], -self.llr_max, self.llr_max)

        msg_l_hist = []
        msg_r_hist = []

        msg_r_in = np.zeros(self.N, dtype=np.float64)
        msg_r_in[self.frozen_pos] = self.llr_max

        num_iters = 0
        u_hat = np.zeros(self.N, dtype=int)
        msg_l = [np.zeros(self.N) for _ in range(self.n + 1)]

        for it in range(self.max_iter):
            msg_r_it = [None] * (self.n + 1)
            msg_l_it = [None] * (self.n + 1)

            for stage in range(self.n):
                ind_1, ind_2, ind_inv = self._stage_indices(stage)

                if stage == self.n - 1:
                    l1_in = llr_dec[ind_1]
                    l2_in = llr_dec[ind_2]
                elif it == 0:
                    l1_in = np.zeros(self.N // 2)
                    l2_in = np.zeros(self.N // 2)
                else:
                    l_in = msg_l_hist[it - 1][stage + 1]
                    l1_in = l_in[ind_1]
                    l2_in = l_in[ind_2]

                if stage == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_in = msg_r_it[stage]
                    r1_in = r_in[ind_1]
                    r2_in = r_in[ind_2]

                r1_out = _f_min_sum(r1_in, l2_in + r2_in, self.alpha)
                r2_out = _f_min_sum(r1_in, l1_in, self.alpha) + r2_in
                r_out = np.concatenate([r1_out, r2_out])[ind_inv]
                msg_r_it[stage + 1] = r_out

            for stage in range(self.n - 1, -1, -1):
                ind_1, ind_2, ind_inv = self._stage_indices(stage)

                if stage == self.n - 1:
                    l1_in = llr_dec[ind_1]
                    l2_in = llr_dec[ind_2]
                else:
                    l_in = msg_l_it[stage + 1]
                    l1_in = l_in[ind_1]
                    l2_in = l_in[ind_2]

                if stage == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_in = msg_r_it[stage]
                    r1_in = r_in[ind_1]
                    r2_in = r_in[ind_2]

                l1_out = _f_min_sum(l1_in, l2_in + r2_in, self.alpha)
                l2_out = _f_min_sum(r1_in, l1_in, self.alpha) + l2_in
                l_out = np.concatenate([l1_out, l2_out])[ind_inv]
                msg_l_it[stage] = l_out

            msg_l_hist.append(msg_l_it)
            msg_r_hist.append(msg_r_it)
            msg_l = msg_l_it
            num_iters = it + 1

            total = msg_l[0] + msg_r_in
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0
            if self._check_early_stop(u_hat, llr_ch):
                break

        total = msg_l[0] + msg_r_in
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
