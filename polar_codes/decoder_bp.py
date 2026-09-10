"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode


def _boxplus(x, y, llr_max=19.3):
    """Sum-product 盒加运算。"""
    x = np.clip(x, -llr_max, llr_max)
    y = np.clip(y, -llr_max, llr_max)
    return np.log1p(np.exp(x + y)) - np.log(np.exp(x) + np.exp(y))


def _minsum_f(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375, use_minsum=False):
        self.N = N
        self.n_stages = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_pos = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.use_minsum = use_minsum
        self._llr_max = 19.3

    def _cn(self, x, y):
        if self.use_minsum:
            return _minsum_f(x, y, self.alpha)
        return _boxplus(x, y, self._llr_max)

    def _stage_indices(self, stage):
        ind_range = np.arange(self.N // 2)
        ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** stage)
        ind_2 = ind_1 + 2 ** stage
        ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
        return ind_1, ind_2, ind_inv

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n_stages
        N = self.N

        msg_r_in = np.zeros(N, dtype=np.float64)
        msg_r_in[self.frozen_pos] = self._llr_max

        msg_l = [None] * (n + 1)
        msg_r = [None] * (n + 1)

        num_iters = self.max_iter
        for ind_it in range(self.max_iter):
            msg_r = [None] * (n + 1)
            for ind_s in range(n):
                ind_1, ind_2, ind_inv = self._stage_indices(ind_s)

                if ind_s == n - 1:
                    l1_in, l2_in = llr_ch[ind_1], llr_ch[ind_2]
                elif ind_it == 0:
                    l1_in = np.zeros(len(ind_1))
                    l2_in = np.zeros(len(ind_2))
                else:
                    l_in = msg_l[ind_s + 1]
                    l1_in, l2_in = l_in[ind_1], l_in[ind_2]

                if ind_s == 0:
                    r1_in, r2_in = msg_r_in[ind_1], msg_r_in[ind_2]
                else:
                    r_in = msg_r[ind_s]
                    r1_in, r2_in = r_in[ind_1], r_in[ind_2]

                r1_out = self._cn(r1_in, l2_in + r2_in)
                r2_out = self._cn(r1_in, l1_in) + r2_in
                r_out = np.concatenate([r1_out, r2_out])[ind_inv]
                msg_r[ind_s + 1] = r_out

            msg_l = [None] * (n + 1)
            for ind_s in range(n - 1, -1, -1):
                ind_1, ind_2, ind_inv = self._stage_indices(ind_s)

                if ind_s == n - 1:
                    l1_in, l2_in = llr_ch[ind_1], llr_ch[ind_2]
                else:
                    l_in = msg_l[ind_s + 1]
                    l1_in, l2_in = l_in[ind_1], l_in[ind_2]

                if ind_s == 0:
                    r1_in, r2_in = msg_r_in[ind_1], msg_r_in[ind_2]
                else:
                    r_in = msg_r[ind_s]
                    r1_in, r2_in = r_in[ind_1], r_in[ind_2]

                l1_out = self._cn(l1_in, l2_in + r2_in)
                l2_out = self._cn(r1_in, l1_in) + l2_in
                msg_l[ind_s] = np.concatenate([l1_out, l2_out])[ind_inv]

            llr_total = msg_l[0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[self.info_idx] = (llr_total[self.info_idx] < 0).astype(int)

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = ind_it + 1
                break

        llr_total = msg_l[0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[self.info_idx] = (llr_total[self.info_idx] < 0).astype(int)
        return u_hat, num_iters
