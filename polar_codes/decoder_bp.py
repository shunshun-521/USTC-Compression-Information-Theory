"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode
from decoder_sc import _reorder_channel_llrs

LLR_MAX = 19.3
LARGE = LLR_MAX


def _clamp(x):
    return np.clip(x, -LLR_MAX, LLR_MAX)


def _f_min_sum(a, b, alpha):
    a = _clamp(a)
    b = _clamp(b)
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（因子图消息传递，配合编码端比特倒序）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def _run_bp(self, llr_ch, num_iter):
        n_stages = self.n
        n = self.N
        alpha = self.alpha

        msg_l = [[None] * (n_stages + 1) for _ in range(num_iter)]
        msg_r = [[None] * (n_stages + 1) for _ in range(num_iter)]
        msg_r_in = np.zeros(n, dtype=np.float64)
        msg_r_in[self.frozen_idx] = LARGE
        llr_ch = _clamp(llr_ch)

        for ind_it in range(num_iter):
            for ind_s in range(n_stages):
                ind_range = np.arange(n // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
                ind_2 = ind_1 + 2 ** ind_s

                if ind_s == n_stages - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                elif ind_it == 0:
                    l1_in = np.zeros(n // 2)
                    l2_in = np.zeros(n // 2)
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

                r1_out = _f_min_sum(r1_in, l2_in + r2_in, alpha)
                r2_out = _f_min_sum(r1_in, l1_in, alpha) + r2_in
                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
                msg_r[ind_it][ind_s + 1] = np.concatenate([r1_out, r2_out])[ind_inv]

            for ind_s in range(n_stages - 1, -1, -1):
                ind_range = np.arange(n // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
                ind_2 = ind_1 + 2 ** ind_s
                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))

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

                l1_out = _f_min_sum(l1_in, l2_in + r2_in, alpha)
                l2_out = _f_min_sum(r1_in, l1_in, alpha) + l2_in
                msg_l[ind_it][ind_s] = np.concatenate([l1_out, l2_out])[ind_inv]

        return msg_l[num_iter - 1][0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_ordered = _reorder_channel_llrs(llr_ch, self.n)
        soft = self._run_bp(llr_ordered, self.max_iter)
        u_hat = (soft < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            soft_it = self._run_bp(llr_ordered, it)
            u_try = (soft_it < 0).astype(int)
            u_try[self.frozen_idx] = 0
            x_hat = polar_encode(u_try)
            hard_ch = (llr_ordered < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                u_hat = u_try
                num_iters = it
                break

        return u_hat, num_iters
