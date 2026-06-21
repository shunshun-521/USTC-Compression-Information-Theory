"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（参考 Sionna/Arikan 因子图结构）。"""

    LLR_MAX = 19.3

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_pos = np.where(self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)

    def _boxplus(self, x, y):
        return self.alpha * f_operation(x, y)

    def _hard_bits(self, llr_stage):
        u_hat = np.zeros(self.N, dtype=int)
        u_hat[~self.frozen_bits] = (llr_stage[~self.frozen_bits] < 0).astype(int)
        return u_hat

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = llr_ch[self.br]
        n = self.N
        n_stages = self.n
        num_iter = self.max_iter
        hard_ch = (llr_ch < 0).astype(int)

        msg_l = [[None] * (n_stages + 1) for _ in range(num_iter)]
        msg_r = [[None] * (n_stages + 1) for _ in range(num_iter)]

        msg_r_in = np.zeros(n, dtype=np.float64)
        msg_r_in[self.frozen_pos] = self.LLR_MAX

        u_hat = np.zeros(n, dtype=int)
        num_iters = num_iter

        for ind_it in range(num_iter):
            for ind_s in range(n_stages):
                ind_range = np.arange(n // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
                ind_2 = ind_1 + 2 ** ind_s

                if ind_s == n_stages - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                elif ind_it == 0:
                    l1_in = np.zeros(n // 2, dtype=np.float64)
                    l2_in = np.zeros(n // 2, dtype=np.float64)
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

                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
                msg_r[ind_it][ind_s + 1] = np.concatenate([r1_out, r2_out])[ind_inv]

            for ind_s in range(n_stages - 1, -1, -1):
                ind_range = np.arange(n // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
                ind_2 = ind_1 + 2 ** ind_s

                if ind_s == n_stages - 1:
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

                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
                msg_l[ind_it][ind_s] = np.concatenate([l1_out, l2_out])[ind_inv]

            u_hat = self._hard_bits(msg_l[ind_it][0])
            if np.array_equal(polar_encode(u_hat), hard_ch):
                num_iters = ind_it + 1
                break
        else:
            u_hat = self._hard_bits(msg_l[num_iter - 1][0])

        return u_hat, num_iters
