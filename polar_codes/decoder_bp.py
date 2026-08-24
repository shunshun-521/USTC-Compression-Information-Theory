"""
极化码 BP（置信传播）译码器
Sionna/Arikan 因子图 flooding schedule + min-sum 近似
输入 LLR 需与标准 F^{⊗n} 因子图对齐（对 B_N 编码做比特倒序置换）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _boxplus_ms(x, y, alpha=0.9375):
    """min-sum 近似的 boxplus"""
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器"""

    LLR_MAX = 19.3

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n_stages = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_pos = np.where(self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self._br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_bp = llr_ch[self._br]

        N = self.N
        n = self.n_stages
        alpha = self.alpha

        msg_r_in = np.zeros(N, dtype=np.float64)
        msg_r_in[self.frozen_pos] = self.LLR_MAX

        msg_l = [None] * (n + 1)
        msg_r = [None] * (n + 1)

        num_iters = self.max_iter

        for ind_it in range(self.max_iter):
            for ind_s in range(n):
                ind_range = np.arange(N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
                ind_2 = ind_1 + 2 ** ind_s

                if ind_s == n - 1:
                    l1_in = llr_bp[ind_1]
                    l2_in = llr_bp[ind_2]
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

                r1_out = _boxplus_ms(r1_in, l2_in + r2_in, alpha)
                r2_out = _boxplus_ms(r1_in, l1_in, alpha) + r2_in

                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
                msg_r[ind_s + 1] = np.concatenate([r1_out, r2_out])[ind_inv]

            for ind_s in range(n - 1, -1, -1):
                ind_range = np.arange(N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
                ind_2 = ind_1 + 2 ** ind_s
                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))

                if ind_s == n - 1:
                    l1_in = llr_bp[ind_1]
                    l2_in = llr_bp[ind_2]
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

                l1_out = _boxplus_ms(l1_in, l2_in + r2_in, alpha)
                l2_out = _boxplus_ms(r1_in, l1_in, alpha) + l2_in

                msg_l[ind_s] = np.concatenate([l1_out, l2_out])[ind_inv]

            u_hat = self._hard_decision(msg_l[0])
            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                num_iters = ind_it + 1
                break

        return self._hard_decision(msg_l[0]), num_iters

    def _hard_decision(self, llr_left):
        u_hat = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if llr_left[i] > 0 else 1
        return u_hat
