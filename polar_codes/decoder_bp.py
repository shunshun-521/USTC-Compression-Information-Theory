"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _boxplus_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（Sionna/Arikan 因子图调度，min-sum 近似）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self._llr_max = 30.0

    def _stage_indices(self, stage):
        ind_range = np.arange(self.N // 2)
        ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** stage)
        ind_2 = ind_1 + 2 ** stage
        ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
        return ind_1, ind_2, ind_inv

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        msg_r_in = np.zeros(N, dtype=np.float64)
        msg_r_in[self.frozen_idx] = self._llr_max

        msg_l = [None] * (n + 1)
        msg_r = [None] * (n + 1)

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(self.max_iter):
            # 左 -> 右更新 R
            for s in range(n):
                ind_1, ind_2, ind_inv = self._stage_indices(s)

                if s == n - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
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

                r1_out = _boxplus_min_sum(r1_in, l2_in + r2_in, self.alpha)
                r2_out = _boxplus_min_sum(r1_in, l1_in, self.alpha) + r2_in
                msg_r[s + 1] = np.concatenate([r1_out, r2_out])[ind_inv]

            # 右 -> 左更新 L
            for s in range(n - 1, -1, -1):
                ind_1, ind_2, ind_inv = self._stage_indices(s)

                if s == n - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
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

                l1_out = _boxplus_min_sum(l1_in, l2_in + r2_in, self.alpha)
                l2_out = _boxplus_min_sum(r1_in, l1_in, self.alpha) + l2_in
                msg_l[s] = np.concatenate([l1_out, l2_out])[ind_inv]

            total = msg_l[0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[self.info_idx] = (total[self.info_idx] < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            num_iters = it + 1
            if np.array_equal(x_hat, hard_ch):
                break

        return u_hat, num_iters
