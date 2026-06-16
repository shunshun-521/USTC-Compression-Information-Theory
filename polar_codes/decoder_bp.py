"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _minsum_f(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.frozen_br = self.br[self.frozen_idx]
        self.info_br = self.br[np.where(self.frozen_bits == 0)[0]]
        self.LARGE = 1e6

    def _stage_indices(self, stage):
        ind_range = np.arange(self.N // 2)
        ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** stage)
        ind_2 = ind_1 + 2 ** stage
        ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
        return ind_1, ind_2, ind_inv

    def _to_natural(self, u_br):
        u_hat = np.zeros(self.N, dtype=int)
        u_hat[self.br] = u_br
        return u_hat

    def decode(self, llr_ch):
        """主译码函数。"""
        N, n = self.N, self.n
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        f = lambda a, b: _minsum_f(a, b, self.alpha)

        msg_l = [None] * (n + 1)
        msg_r = [None] * (n + 1)

        msg_r_in = np.zeros(N, dtype=np.float64)
        msg_r_in[self.frozen_br] = self.LARGE

        num_iters = 0
        u_br = np.zeros(N, dtype=int)

        for it in range(self.max_iter):
            for stage in range(n):
                ind_1, ind_2, ind_inv = self._stage_indices(stage)

                if stage == n - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                elif it == 0:
                    l1_in = np.zeros(len(ind_1))
                    l2_in = np.zeros(len(ind_2))
                else:
                    l1_in = msg_l[stage + 1][ind_1]
                    l2_in = msg_l[stage + 1][ind_2]

                if stage == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r1_in = msg_r[stage][ind_1]
                    r2_in = msg_r[stage][ind_2]

                r1_out = f(r1_in, l2_in + r2_in)
                r2_out = f(r1_in, l1_in) + r2_in
                msg_r[stage + 1] = np.concatenate([r1_out, r2_out])[ind_inv]

            for stage in range(n - 1, -1, -1):
                ind_1, ind_2, ind_inv = self._stage_indices(stage)

                if stage == n - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                else:
                    l1_in = msg_l[stage + 1][ind_1]
                    l2_in = msg_l[stage + 1][ind_2]

                if stage == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r1_in = msg_r[stage][ind_1]
                    r2_in = msg_r[stage][ind_2]

                l1_out = f(l1_in, l2_in + r2_in)
                l2_out = f(r1_in, l1_in) + l2_in
                msg_l[stage] = np.concatenate([l1_out, l2_out])[ind_inv]

            soft = msg_l[0]
            u_br[:] = 0
            u_br[self.info_br] = (soft[self.info_br] < 0).astype(int)

            u_hat = self._to_natural(u_br)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            num_iters = it + 1
            if np.array_equal(x_hat, hard_ch):
                break

        u_hat = self._to_natural(u_br)
        return u_hat, num_iters
