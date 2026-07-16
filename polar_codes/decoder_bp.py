"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode, bit_reversal_permutation


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


def _stage_indices(N, ind_s):
    ind_range = np.arange(N // 2)
    ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
    ind_2 = ind_1 + 2 ** ind_s
    ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
    return ind_1, ind_2, ind_inv


class BPDecoder:
    """BP 译码器（基于 Sionna/Arikan 因子图结构）"""

    LLR_MAX = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self._stage_cache = [_stage_indices(N, s) for s in range(self.n)]

    def _hard_decision(self, llr_vec):
        u_hat = (llr_vec < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        rev = bit_reversal_permutation(self.N)
        llr_ch = llr_ch[rev]

        N, n_stages = self.N, self.n
        alpha = self.alpha

        msg_r_in = np.zeros(N, dtype=np.float64)
        msg_r_in[self.frozen_idx] = self.LLR_MAX

        msg_l_prev = None
        num_iters = 0

        for it in range(self.max_iter):
            msg_l = [None] * (n_stages + 1)
            msg_r = [None] * (n_stages + 1)

            for ind_s in range(n_stages):
                ind_1, ind_2, ind_inv = self._stage_cache[ind_s]

                if ind_s == n_stages - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                elif it == 0:
                    l1_in = np.zeros(N // 2)
                    l2_in = np.zeros(N // 2)
                else:
                    l_in = msg_l_prev[ind_s + 1]
                    l1_in = l_in[ind_1]
                    l2_in = l_in[ind_2]

                if ind_s == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_in = msg_r[ind_s]
                    r1_in = r_in[ind_1]
                    r2_in = r_in[ind_2]

                r1_out = _f_min_sum(r1_in, l2_in + r2_in, alpha)
                r2_out = _f_min_sum(r1_in, l1_in, alpha) + r2_in
                msg_r[ind_s + 1] = np.concatenate([r1_out, r2_out])[ind_inv]

            for ind_s in range(n_stages - 1, -1, -1):
                ind_1, ind_2, ind_inv = self._stage_cache[ind_s]

                if ind_s == n_stages - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                else:
                    l_in = msg_l[ind_s + 1]
                    l1_in = l_in[ind_1]
                    l2_in = l_in[ind_2]

                if ind_s == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_in = msg_r[ind_s]
                    r1_in = r_in[ind_1]
                    r2_in = r_in[ind_2]

                l1_out = _f_min_sum(l1_in, l2_in + r2_in, alpha)
                l2_out = _f_min_sum(r1_in, l1_in, alpha) + l2_in
                msg_l[ind_s] = np.concatenate([l1_out, l2_out])[ind_inv]

            msg_l_prev = msg_l
            num_iters = it + 1

            u_hat = self._hard_decision(msg_l[0])
            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, (llr_ch < 0).astype(int)):
                break

        u_hat = self._hard_decision(msg_l_prev[0])
        return u_hat, num_iters
