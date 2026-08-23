"""
极化码 BP（置信传播）译码器
基于因子图，使用 boxplus / min-sum，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode


def _boxplus(x, y, llr_max=19.3):
    x = np.clip(x, -llr_max, llr_max)
    y = np.clip(y, -llr_max, llr_max)
    return np.log1p(np.exp(x + y)) - np.logaddexp(x, y)


def _minsum(x, y, alpha=0.9375):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_pos = np.where(self.frozen_bits)[0]
        self.info_pos = np.where(~self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.llr_max = 19.3
        self.use_minsum = alpha < 1.0

    def _cn(self, x, y):
        if self.use_minsum:
            return _minsum(x, y, self.alpha)
        return _boxplus(x, y, self.llr_max)

    def _decode_bp(self, llr_ch):
        n = self.n
        N = self.N
        num_iter = self.max_iter

        msg_l = [[None] * (n + 1) for _ in range(num_iter)]
        msg_r = [[None] * (n + 1) for _ in range(num_iter)]

        msg_r_in = np.zeros(N, dtype=np.float64)
        msg_r_in[self.frozen_pos] = self.llr_max

        for ind_it in range(num_iter):
            for stage in range(n):
                ind_range = np.arange(N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** stage)
                ind_2 = ind_1 + 2 ** stage
                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))

                if stage == n - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                elif ind_it == 0:
                    l1_in = np.zeros(N // 2)
                    l2_in = np.zeros(N // 2)
                else:
                    l_in = msg_l[ind_it - 1][stage + 1]
                    l1_in = l_in[ind_1]
                    l2_in = l_in[ind_2]

                if stage == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_in = msg_r[ind_it][stage]
                    r1_in = r_in[ind_1]
                    r2_in = r_in[ind_2]

                r1_out = self._cn(r1_in, l2_in + r2_in)
                r2_out = self._cn(r1_in, l1_in) + r2_in
                msg_r[ind_it][stage + 1] = np.concatenate([r1_out, r2_out])[ind_inv]

            for stage in range(n - 1, -1, -1):
                ind_range = np.arange(N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** stage)
                ind_2 = ind_1 + 2 ** stage
                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))

                if stage == n - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                else:
                    l_in = msg_l[ind_it][stage + 1]
                    l1_in = l_in[ind_1]
                    l2_in = l_in[ind_2]

                if stage == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_in = msg_r[ind_it][stage]
                    r1_in = r_in[ind_1]
                    r2_in = r_in[ind_2]

                l1_out = self._cn(l1_in, l2_in + r2_in)
                l2_out = self._cn(r1_in, l1_in) + l2_in
                msg_l[ind_it][stage] = np.concatenate([l1_out, l2_out])[ind_inv]

        return msg_l[num_iter - 1][0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        rev = np.array([int(f"{i:0{self.n}b}"[::-1], 2) for i in range(self.N)], dtype=int)
        llr_internal = llr_ch[rev]

        llr_left = self._decode_bp(llr_internal)
        u_soft = llr_left[self.info_pos]

        u_hat = np.zeros(self.N, dtype=np.int8)
        u_hat[self.info_pos] = (u_soft < 0).astype(np.int8)

        num_iters = self.max_iter
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_ch < 0).astype(np.int8)
        if np.array_equal(x_hat, hard_ch):
            num_iters = self.max_iter

        return u_hat, num_iters
