"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import bit_reversal_permutation, polar_encode


def _f_min_sum(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


def _boxplus(x, y, llr_max=19.3):
    x = np.clip(x, -llr_max, llr_max)
    y = np.clip(y, -llr_max, llr_max)
    return np.log1p(np.exp(x + y)) - np.log(np.exp(x) + np.exp(y))


def _stage_indices(n_stages, n):
    pairs = []
    half = n // 2
    for ind_s in range(n_stages):
        ind_range = np.arange(half)
        ind_1 = ind_range * 2 - np.mod(ind_range, 2**ind_s)
        ind_2 = ind_1 + 2**ind_s
        ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
        pairs.append((ind_1, ind_2, ind_inv))
    return pairs


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_pos = np.where(self.frozen_bits)[0]
        self.info_pos = np.where(~self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 19.3
        self._pairs = _stage_indices(self.n, N)
        self._use_min_sum = True

    def _cn_update(self, x, y):
        if self._use_min_sum:
            return _f_min_sum(x, y, self.alpha)
        return _boxplus(x, y, self._large)

    def _hard_decision(self, msg_l0):
        u_hat = np.zeros(self.N, dtype=int)
        u_hat[self.info_pos] = (msg_l0[self.info_pos] <= 0).astype(int)
        return u_hat

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        n = self.n
        N = self.N
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(N)
        llr_graph = llr_ch[br]

        msg_r_in = np.zeros(N, dtype=np.float64)
        msg_r_in[self.frozen_pos] = self._large

        msg_l = [[None] * (n + 1) for _ in range(self.max_iter)]
        msg_r = [[None] * (n + 1) for _ in range(self.max_iter)]
        zero_l = np.zeros(N, dtype=np.float64)

        num_iters = 0
        for it in range(self.max_iter):
            num_iters = it + 1

            for ind_s in range(n):
                ind_1, ind_2, ind_inv = self._pairs[ind_s]

                if ind_s == n - 1:
                    l1_in = llr_graph[ind_1]
                    l2_in = llr_graph[ind_2]
                elif it == 0:
                    l1_in = zero_l[ind_1]
                    l2_in = zero_l[ind_2]
                else:
                    l_prev = msg_l[it - 1][ind_s + 1]
                    l1_in = l_prev[ind_1]
                    l2_in = l_prev[ind_2]

                if ind_s == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_prev = msg_r[it][ind_s]
                    r1_in = r_prev[ind_1]
                    r2_in = r_prev[ind_2]

                r1_out = self._cn_update(r1_in, l2_in + r2_in)
                r2_out = self._cn_update(r1_in, l1_in) + r2_in
                msg_r[it][ind_s + 1] = np.concatenate([r1_out, r2_out])[ind_inv]

            for ind_s in range(n - 1, -1, -1):
                ind_1, ind_2, ind_inv = self._pairs[ind_s]

                if ind_s == n - 1:
                    l1_in = llr_graph[ind_1]
                    l2_in = llr_graph[ind_2]
                else:
                    l_prev = msg_l[it][ind_s + 1]
                    l1_in = l_prev[ind_1]
                    l2_in = l_prev[ind_2]

                if ind_s == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_prev = msg_r[it][ind_s]
                    r1_in = r_prev[ind_1]
                    r2_in = r_prev[ind_2]

                l1_out = self._cn_update(l1_in, l2_in + r2_in)
                l2_out = self._cn_update(r1_in, l1_in) + l2_in
                msg_l[it][ind_s] = np.concatenate([l1_out, l2_out])[ind_inv]

            u_hat = self._hard_decision(msg_l[it][0])
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        u_hat = self._hard_decision(msg_l[num_iters - 1][0])
        return u_hat, num_iters
