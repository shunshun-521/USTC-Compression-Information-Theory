"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation


def _boxplus(x, y, llr_max=19.3):
    """Check-node update (sum-product box-plus)。"""
    x = np.clip(x, -llr_max, llr_max)
    y = np.clip(y, -llr_max, llr_max)
    return np.log1p(np.exp(x + y)) - np.log(np.exp(x) + np.exp(y))


class BPDecoder:
    """BP 译码器（参考 Sionna PolarBPDecoder 结构）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n_stages = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_pos = np.where(self.frozen_bits == 1)[0]
        self.info_pos = np.where(self.frozen_bits == 0)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.llr_max = 19.3

    def _f(self, x, y):
        """Scaled min-sum，用于可选替代 boxplus。"""
        from decoder_sc import f_operation
        x = np.clip(x, -self.llr_max, self.llr_max)
        y = np.clip(y, -self.llr_max, self.llr_max)
        zero_x = np.abs(x) < 1e-12
        zero_y = np.abs(y) < 1e-12
        both = ~(zero_x | zero_y)
        out = np.zeros_like(x, dtype=np.float64)
        out[zero_x] = y[zero_x]
        out[zero_y] = x[zero_y]
        if np.any(both):
            out[both] = self.alpha * f_operation(x[both], y[both])
        return out

    def _cn(self, x, y):
        return _boxplus(x, y, self.llr_max)

    def decode(self, llr_ch):
        llr_ch = np.clip(np.asarray(llr_ch, dtype=np.float64), -self.llr_max, self.llr_max)
        br = bit_reversal_permutation(self.N)
        llr_ch = llr_ch[br]
        N = self.N
        n_stages = self.n_stages
        num_iter = self.max_iter

        msg_l = [[None] * (n_stages + 1) for _ in range(num_iter)]
        msg_r = [[None] * (n_stages + 1) for _ in range(num_iter)]

        msg_r_in = np.zeros(N, dtype=np.float64)
        msg_r_in[self.frozen_pos] = self.llr_max

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for ind_it in range(num_iter):
            num_iters = ind_it + 1

            for ind_s in range(n_stages):
                ind_range = np.arange(N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
                ind_2 = ind_1 + 2 ** ind_s

                if ind_s == n_stages - 1:
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

                r1_out = self._cn(r1_in, l2_in + r2_in)
                r2_out = self._cn(r1_in, l1_in) + r2_in

                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
                r_out = np.concatenate([r1_out, r2_out])[ind_inv]
                msg_r[ind_it][ind_s + 1] = r_out

            for ind_s in range(n_stages - 1, -1, -1):
                ind_range = np.arange(N // 2)
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

                l1_out = self._cn(l1_in, l2_in + r2_in)
                l2_out = self._cn(r1_in, l1_in) + l2_in
                l_out = np.concatenate([l1_out, l2_out])[ind_inv]
                msg_l[ind_it][ind_s] = l_out

            total = msg_l[ind_it][0]
            for i in range(N):
                u_hat[i] = 0 if self.frozen_bits[i] else (0 if total[i] > 0 else 1)

            x_hat = polar_encode(u_hat)
            hard_bits = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_bits):
                break

        total = msg_l[num_iters - 1][0]
        for i in range(N):
            u_hat[i] = 0 if self.frozen_bits[i] else (0 if total[i] > 0 else 1)
        return u_hat, num_iters
