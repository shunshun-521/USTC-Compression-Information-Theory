"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
实现参考 Sionna PolarBPDecoder 的因子图索引与消息传递结构。
"""
import math

import numpy as np

from decoder_sc import _logdomain_sum, f_operation_min_sum
from encoder import polar_encode, prepare_channel_llr

LLR_MAX = 19.3


class BPDecoder:
    """
    BP 译码器。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375, use_min_sum=True):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.use_min_sum = use_min_sum

    def _boxplus(self, x, y):
        if self.use_min_sum:
            return self.alpha * f_operation_min_sum(x, y)
        x = np.clip(x, -LLR_MAX, LLR_MAX)
        y = np.clip(y, -LLR_MAX, LLR_MAX)
        return _logdomain_sum(x + y, 0.0) - _logdomain_sum(x, y)

    def decode(self, llr_ch):
        llr_ch = prepare_channel_llr(np.asarray(llr_ch, dtype=np.float64))
        n = self.n
        N = self.N
        num_iter = self.max_iter

        msg_l = [[None] * (n + 1) for _ in range(num_iter)]
        msg_r = [[None] * (n + 1) for _ in range(num_iter)]

        msg_r_in = np.zeros(N, dtype=np.float64)
        msg_r_in[self.frozen_idx] = LLR_MAX

        u_hat = np.zeros(N, dtype=int)
        actual_iters = 0

        for ind_it in range(num_iter):
            actual_iters = ind_it + 1

            for ind_s in range(n):
                ind_range = np.arange(N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
                ind_2 = ind_1 + 2 ** ind_s

                if ind_s == n - 1:
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

                r1_out = self._boxplus(r1_in, l2_in + r2_in)
                r2_out = self._boxplus(r1_in, l1_in) + r2_in

                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
                msg_r[ind_it][ind_s + 1] = np.concatenate([r1_out, r2_out])[ind_inv]

            for ind_s in range(n - 1, -1, -1):
                ind_range = np.arange(N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
                ind_2 = ind_1 + 2 ** ind_s
                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))

                if ind_s == n - 1:
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
                msg_l[ind_it][ind_s] = np.concatenate([l1_out, l2_out])[ind_inv]

            llr_u = msg_l[ind_it][0]
            for i in range(N):
                u_hat[i] = 0 if self.frozen_bits[i] else (0 if llr_u[i] >= 0 else 1)

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, (llr_ch < 0).astype(int)):
                break

        llr_u = msg_l[actual_iters - 1][0]
        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if llr_u[i] >= 0 else 1

        return u_hat, actual_iters
