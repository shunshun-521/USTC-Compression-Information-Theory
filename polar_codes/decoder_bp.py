"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode
from decoder_sc import _prepare_llr


class BPDecoder:
    """BP 译码器（参考 Sionna/Arikan 因子图索引）。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_pos = np.where(self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha

    def _f_min_sum(self, x, y):
        return self.alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr, _ = _prepare_llr(llr_ch)
        n = self.n
        N = self.N
        num_iter = self.max_iter

        msg_r_in = np.zeros(N, dtype=np.float64)
        msg_r_in[self.frozen_pos] = self.LARGE

        msg_l = [None] * (n + 1)
        msg_r = [None] * (n + 1)

        for it in range(num_iter):
            msg_r = [None] * (n + 1)
            msg_r[0] = msg_r_in.copy()

            for s in range(n):
                ind_range = np.arange(N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** s)
                ind_2 = ind_1 + 2 ** s

                if s == n - 1:
                    l1_in = llr[ind_1]
                    l2_in = llr[ind_2]
                elif it == 0:
                    l1_in = np.zeros(N // 2)
                    l2_in = np.zeros(N // 2)
                else:
                    l_prev = msg_l[s + 1]
                    l1_in = l_prev[ind_1]
                    l2_in = l_prev[ind_2]

                if s == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_prev = msg_r[s]
                    r1_in = r_prev[ind_1]
                    r2_in = r_prev[ind_2]

                r1_out = self._f_min_sum(r1_in, l2_in + r2_in)
                r2_out = self._f_min_sum(r1_in, l1_in) + r2_in

                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
                msg_r[s + 1] = np.concatenate([r1_out, r2_out])[ind_inv]

            msg_l = [None] * (n + 1)
            for s in range(n - 1, -1, -1):
                ind_range = np.arange(N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** s)
                ind_2 = ind_1 + 2 ** s
                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))

                if s == n - 1:
                    l1_in = llr[ind_1]
                    l2_in = llr[ind_2]
                else:
                    l_prev = msg_l[s + 1]
                    l1_in = l_prev[ind_1]
                    l2_in = l_prev[ind_2]

                if s == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_prev = msg_r[s]
                    r1_in = r_prev[ind_1]
                    r2_in = r_prev[ind_2]

                l1_out = self._f_min_sum(l1_in, l2_in + r2_in)
                l2_out = self._f_min_sum(r1_in, l1_in) + l2_in
                msg_l[s] = np.concatenate([l1_out, l2_out])[ind_inv]

            total = msg_l[0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iter = it + 1
                return u_hat, num_iter

        total = msg_l[0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iter
