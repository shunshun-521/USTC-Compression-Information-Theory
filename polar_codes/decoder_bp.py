"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math

from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（基于 Arikan 因子图，min-sum 近似）。"""

    LLR_MAX = 19.3

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n_stages = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.info_idx = np.where(self.frozen_bits == 0)[0]

    def _ms_boxplus(self, x, y):
        return self.alpha * f_operation(
            np.clip(x, -self.LLR_MAX, self.LLR_MAX),
            np.clip(y, -self.LLR_MAX, self.LLR_MAX),
        )

    def _get_indices(self, stage):
        ind_range = np.arange(self.N // 2)
        ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** stage)
        ind_2 = ind_1 + 2 ** stage
        ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
        return ind_1, ind_2, ind_inv

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n_stages = self.n_stages
        N = self.N

        msg_r_in = np.zeros(N, dtype=np.float64)
        msg_r_in[self.frozen_idx] = self.LLR_MAX

        msg_l = [None] * (n_stages + 1)
        msg_r = [None] * (n_stages + 1)

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for ind_it in range(self.max_iter):
            for ind_s in range(n_stages):
                ind_1, ind_2, ind_inv = self._get_indices(ind_s)

                if ind_s == n_stages - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                elif ind_it == 0:
                    l1_in = np.zeros(N // 2)
                    l2_in = np.zeros(N // 2)
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

                r1_out = self._ms_boxplus(r1_in, l2_in + r2_in)
                r2_out = self._ms_boxplus(r1_in, l1_in) + r2_in
                r_out = np.concatenate([r1_out, r2_out])[ind_inv]
                msg_r[ind_s + 1] = r_out

            for ind_s in range(n_stages - 1, -1, -1):
                ind_1, ind_2, ind_inv = self._get_indices(ind_s)

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

                l1_out = self._ms_boxplus(l1_in, l2_in + r2_in)
                l2_out = self._ms_boxplus(r1_in, l1_in) + l2_in
                l_out = np.concatenate([l1_out, l2_out])[ind_inv]
                msg_l[ind_s] = l_out

            num_iters = ind_it + 1
            llr_total = msg_l[0]
            for i in range(N):
                u_hat[i] = 0 if llr_total[i] >= 0 else 1
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        llr_total = msg_l[0]
        u_hat_full = np.zeros(N, dtype=int)
        for i in range(N):
            u_hat_full[i] = 0 if llr_total[i] >= 0 else 1
        u_hat_full[self.frozen_idx] = 0

        return u_hat_full, num_iters
