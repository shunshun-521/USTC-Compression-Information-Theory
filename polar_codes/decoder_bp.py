"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _boxplus(x, y, alpha):
    """min-sum box-plus。"""
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


def _scatter_outputs(ind_1, ind_2, out_1, out_2, n):
    merged = np.concatenate([out_1, out_2])
    indices = np.concatenate([ind_1, ind_2])
    order = np.argsort(indices)
    result = np.zeros(n, dtype=np.float64)
    result[indices[order]] = merged[order]
    return result


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n_stages = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]
        self.llr_max = 1e6
        self.rev = bit_reversal_permutation(N)

    def _get_indices(self, stage):
        ind_range = np.arange(self.N // 2)
        ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** stage)
        ind_2 = ind_1 + 2 ** stage
        valid = ind_2 < self.N
        return ind_1[valid], ind_2[valid]

    def _iterate(self, llr_v, llr_ch):
        msg_r_in = np.zeros(self.N, dtype=np.float64)
        msg_r_in[self.frozen_idx] = self.llr_max
        msg_l_prev = None

        for it in range(self.max_iter):
            msg_r = [None] * (self.n_stages + 1)
            for stage in range(self.n_stages):
                ind_1, ind_2 = self._get_indices(stage)
                if stage == self.n_stages - 1:
                    l1_in, l2_in = llr_v[ind_1], llr_v[ind_2]
                elif it == 0:
                    l1_in = np.zeros(len(ind_1))
                    l2_in = np.zeros(len(ind_2))
                else:
                    l_in = msg_l_prev[stage + 1]
                    l1_in, l2_in = l_in[ind_1], l_in[ind_2]

                if stage == 0:
                    r1_in, r2_in = msg_r_in[ind_1], msg_r_in[ind_2]
                else:
                    r_in = msg_r[stage]
                    r1_in, r2_in = r_in[ind_1], r_in[ind_2]

                r1_out = _boxplus(r1_in, l2_in + r2_in, self.alpha)
                r2_out = _boxplus(r1_in, l1_in, self.alpha) + r2_in
                msg_r[stage + 1] = _scatter_outputs(ind_1, ind_2, r1_out, r2_out, self.N)

            msg_l = [None] * (self.n_stages + 1)
            for stage in range(self.n_stages - 1, -1, -1):
                ind_1, ind_2 = self._get_indices(stage)
                if stage == self.n_stages - 1:
                    l1_in, l2_in = llr_v[ind_1], llr_v[ind_2]
                else:
                    l_in = msg_l[stage + 1]
                    l1_in, l2_in = l_in[ind_1], l_in[ind_2]

                if stage == 0:
                    r1_in, r2_in = msg_r_in[ind_1], msg_r_in[ind_2]
                else:
                    r_in = msg_r[stage]
                    r1_in, r2_in = r_in[ind_1], r_in[ind_2]

                l1_out = _boxplus(l1_in, l2_in + r2_in, self.alpha)
                l2_out = _boxplus(r1_in, l1_in, self.alpha) + l2_in
                msg_l[stage] = _scatter_outputs(ind_1, ind_2, l1_out, l2_out, self.N)

            msg_l_prev = msg_l
            u_hat = (msg_l[0] < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                return u_hat, it + 1

        u_hat = (msg_l_prev[0] < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, self.max_iter

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_v = llr_ch[self.rev]
        u_hat, num_iters = self._iterate(llr_v, llr_ch)
        return u_hat, num_iters
