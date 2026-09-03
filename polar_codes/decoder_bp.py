"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
（算法参考 Sionna PolarBPDecoder / Arikan BP）
"""
import math
import numpy as np

from encoder import polar_encode, bit_reversal_permutation
from channel import hard_decision_llr


def _boxplus(x, y, llr_max=19.3):
    x = np.clip(x, -llr_max, llr_max)
    y = np.clip(y, -llr_max, llr_max)
    return np.log1p(np.exp(x + y)) - np.log(np.exp(x) + np.exp(y))


def _f_minsum(x, y, alpha=0.9375):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375, use_minsum=True):
        self.N = N
        self.n_stages = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_pos = np.where(self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.use_minsum = use_minsum
        self.llr_max = 19.3

    def _combine(self, a, b):
        if self.use_minsum:
            return _f_minsum(a, b, self.alpha)
        return _boxplus(a, b, self.llr_max)

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, num_iters"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n_s = self.n_stages

        msg_l = [[None] * (n_s + 1) for _ in range(self.max_iter)]
        msg_r = [[None] * (n_s + 1) for _ in range(self.max_iter)]

        msg_r_in = np.zeros(N, dtype=np.float64)
        msg_r_in[self.frozen_pos] = self.llr_max

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int8)

        for ind_it in range(self.max_iter):
            num_iters = ind_it + 1
            for ind_s in range(n_s):
                ind_range = np.arange(N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
                ind_2 = ind_1 + 2 ** ind_s

                if ind_s == n_s - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
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

                r1_out = self._combine(r1_in, l2_in + r2_in)
                r2_out = self._combine(r1_in, l1_in) + r2_in

                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
                r_out = np.concatenate([r1_out, r2_out])[ind_inv]
                msg_r[ind_it][ind_s + 1] = r_out

            for ind_s in range(n_s - 1, -1, -1):
                ind_range = np.arange(N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
                ind_2 = ind_1 + 2 ** ind_s
                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))

                if ind_s == n_s - 1:
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

                l1_out = self._combine(l1_in, l2_in + r2_in)
                l2_out = self._combine(r1_in, l1_in) + l2_in
                l_out = np.concatenate([l1_out, l2_out])[ind_inv]
                msg_l[ind_it][ind_s] = l_out

            total = msg_l[ind_it][0]
            u_hat = (total > 0).astype(np.int8)
            u_hat[self.frozen_pos] = 0

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                break

        total = msg_l[num_iters - 1][0]
        u_hat = (total > 0).astype(np.int8)
        u_hat[self.frozen_pos] = 0
        return u_hat.astype(int), num_iters
