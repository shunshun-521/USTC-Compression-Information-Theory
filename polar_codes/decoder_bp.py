"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _boxplus_minsum(a, b, alpha=0.9375):
    """min-sum 近似 box-plus。"""
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（参考 Sionna/Arikan 因子图结构）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self._llr_max = 19.3

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        msg_r_in = np.zeros(N, dtype=np.float64)
        msg_r_in[self.frozen_idx] = self._llr_max

        msg_l = [None] * (n + 1)
        msg_r = [None] * (n + 1)

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(self.max_iter):
            for s in range(n):
                ind_range = np.arange(N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** s)
                ind_2 = ind_1 + 2 ** s

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

                r1_out = _boxplus_minsum(r1_in, l2_in + r2_in, self.alpha)
                r2_out = _boxplus_minsum(r1_in, l1_in, self.alpha) + r2_in

                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
                r_out = np.concatenate([r1_out, r2_out])[ind_inv]
                msg_r[s + 1] = r_out

            for s in range(n - 1, -1, -1):
                ind_range = np.arange(N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** s)
                ind_2 = ind_1 + 2 ** s
                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))

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

                l1_out = _boxplus_minsum(l1_in, l2_in + r2_in, self.alpha)
                l2_out = _boxplus_minsum(r1_in, l1_in, self.alpha) + l2_in

                l_out = np.concatenate([l1_out, l2_out])[ind_inv]
                msg_l[s] = l_out

            soft = msg_l[0]
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if soft[i] >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it + 1
                break

        soft = msg_l[0]
        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if soft[i] >= 0 else 1

        return u_hat, num_iters
