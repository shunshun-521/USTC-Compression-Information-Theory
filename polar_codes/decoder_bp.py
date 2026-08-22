"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _minsum_boxplus(x, y, alpha):
    x = np.clip(x, -1e6, 1e6)
    y = np.clip(y, -1e6, 1e6)
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器（Sionna 因子图调度 + min-sum）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_pos = np.where(self.frozen_bits == 1)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.llr_max = 19.3

    def _hard_decision(self, llr_vec):
        u_hat = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if llr_vec[i] >= 0 else 1
        return u_hat

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        msg_r_in = np.zeros(N, dtype=np.float64)
        msg_r_in[self.frozen_pos] = self.llr_max

        prev_msg_l = None
        num_iters = self.max_iter
        final_llr = None

        for it in range(self.max_iter):
            msg_r = [None] * (n + 1)
            msg_l = [None] * (n + 1)

            for stage in range(n):
                ind_range = np.arange(N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2**stage)
                ind_2 = ind_1 + 2**stage

                if stage == n - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                elif prev_msg_l is None:
                    l1_in = np.zeros(N // 2)
                    l2_in = np.zeros(N // 2)
                else:
                    l_in = prev_msg_l[stage + 1]
                    l1_in = l_in[ind_1]
                    l2_in = l_in[ind_2]

                if stage == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_in = msg_r[stage]
                    r1_in = r_in[ind_1]
                    r2_in = r_in[ind_2]

                r1_out = _minsum_boxplus(r1_in, l2_in + r2_in, self.alpha)
                r2_out = _minsum_boxplus(r1_in, l1_in, self.alpha) + r2_in

                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
                msg_r[stage + 1] = np.concatenate([r1_out, r2_out])[ind_inv]

            for stage in range(n - 1, -1, -1):
                ind_range = np.arange(N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2**stage)
                ind_2 = ind_1 + 2**stage
                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))

                if stage == n - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                else:
                    l_in = msg_l[stage + 1]
                    l1_in = l_in[ind_1]
                    l2_in = l_in[ind_2]

                if stage == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_in = msg_r[stage]
                    r1_in = r_in[ind_1]
                    r2_in = r_in[ind_2]

                l1_out = _minsum_boxplus(l1_in, l2_in + r2_in, self.alpha)
                l2_out = _minsum_boxplus(r1_in, l1_in, self.alpha) + l2_in
                msg_l[stage] = np.concatenate([l1_out, l2_out])[ind_inv]

            prev_msg_l = msg_l
            final_llr = msg_l[0]
            u_hat = self._hard_decision(final_llr)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it + 1
                break

        u_hat = self._hard_decision(final_llr)
        return u_hat, num_iters
