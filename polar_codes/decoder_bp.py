"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation, _reorder_channel_llrs


def _boxplus(x, y, llr_max=19.3):
    """Sum-product box-plus（数值稳定版）。"""
    x = np.clip(x, -llr_max, llr_max)
    y = np.clip(y, -llr_max, llr_max)
    return np.log1p(np.exp(x + y)) - np.log(np.exp(x) + np.exp(y) + 1e-300)


def _ms_f(x, y, alpha=0.9375):
  return alpha * f_operation(x, y)


class BPDecoder:
    """BP 译码器（基于 Sionna/Arikan 因子图调度）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375, use_min_sum=True):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_pos = np.where(self.frozen_bits)[0]
        self.info_pos = np.where(~self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.use_min_sum = use_min_sum
        self.llr_max = 19.3

    def _combine(self, x, y):
        if self.use_min_sum:
            return _ms_f(x, y, self.alpha)
        return _boxplus(x, y, self.llr_max)

    def decode(self, llr_ch):
        llr_received = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = _reorder_channel_llrs(llr_received)
        n = self.n
        N = self.N
        num_iter = self.max_iter

        msg_r_in = np.zeros(N, dtype=np.float64)
        msg_r_in[self.frozen_pos] = self.llr_max

        msg_l = [None] * (n + 1)
        msg_r = [None] * (n + 1)

        for it in range(num_iter):
            for s in range(n):
                ind_range = np.arange(N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** s)
                ind_2 = ind_1 + 2 ** s
                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))

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

                r1_out = self._combine(r1_in, l2_in + r2_in)
                r2_out = self._combine(r1_in, l1_in) + r2_in
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

                l1_out = self._combine(l1_in, l2_in + r2_in)
                l2_out = self._combine(r1_in, l1_in) + l2_in
                l_out = np.concatenate([l1_out, l2_out])[ind_inv]
                msg_l[s] = l_out

            u_full = np.zeros(N, dtype=int)
            llr_est = msg_l[0]
            u_full[self.info_pos] = (llr_est[self.info_pos] < 0).astype(int)

            if self._early_stop(u_full, llr_received):
                return u_full, it + 1

        u_full = np.zeros(N, dtype=int)
        llr_est = msg_l[0]
        u_full[self.info_pos] = (llr_est[self.info_pos] < 0).astype(int)
        return u_full, num_iter

    def _early_stop(self, u_hat, llr_received):
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_received < 0).astype(int)
        return np.array_equal(x_hat, hard_ch)
