"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from channel import hard_decision_llr
from encoder import bit_reversal_permutation, polar_encode


def _ms_g(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器（参考 Sionna/Arikan 因子图索引，min-sum 近似）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self._large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        alpha = self.alpha
        br = bit_reversal_permutation(N)
        llr = llr_ch[br]

        msg_r_in = np.zeros(N, dtype=np.float64)
        msg_r_in[self.frozen_idx] = self._large

        msg_l = [None] * (n + 1)
        msg_r = [None] * (n + 1)

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(self.max_iter):
            # 左到右：R 更新
            for s in range(n):
                ind_range = np.arange(N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2**s)
                ind_2 = ind_1 + 2**s

                if s == n - 1:
                    l1_in = llr[ind_1]
                    l2_in = llr[ind_2]
                elif it == 0:
                    l1_in = np.zeros(N // 2)
                    l2_in = np.zeros(N // 2)
                else:
                    l_in = msg_l[s + 1]
                    l1_in = l_in[ind_1]
                    l2_in = l_in[ind_2]

                if s == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_in = msg_r[s]
                    r1_in = r_in[ind_1]
                    r2_in = r_in[ind_2]

                r1_out = _ms_g(r1_in, l2_in + r2_in, alpha)
                r2_out = _ms_g(r1_in, l1_in, alpha) + r2_in

                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
                r_out = np.concatenate([r1_out, r2_out])[ind_inv]
                msg_r[s + 1] = r_out

            # 右到左：L 更新
            for s in range(n - 1, -1, -1):
                ind_range = np.arange(N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2**s)
                ind_2 = ind_1 + 2**s
                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))

                if s == n - 1:
                    l1_in = llr[ind_1]
                    l2_in = llr[ind_2]
                else:
                    l_in = msg_l[s + 1]
                    l1_in = l_in[ind_1]
                    l2_in = l_in[ind_2]

                if s == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_in = msg_r[s]
                    r1_in = r_in[ind_1]
                    r2_in = r_in[ind_2]

                l1_out = _ms_g(l1_in, l2_in + r2_in, alpha)
                l2_out = _ms_g(r1_in, l1_in, alpha) + l2_in
                l_out = np.concatenate([l1_out, l2_out])[ind_inv]
                msg_l[s] = l_out

            total = msg_l[0] + msg_r[n]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_decision_llr(llr_ch)):
                num_iters = it + 1
                break

        total = msg_l[0] + msg_r[n]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
