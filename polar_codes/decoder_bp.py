"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation, _to_frozen_mask


class BPDecoder:
    """BP 译码器（参考 Sionna/Arikan 因子图结构）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = _to_frozen_mask(frozen_bits)
        self.frozen_pos = np.where(self.frozen_bits)[0]
        self.info_pos = np.where(~self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.llr_max = 19.3

    def _boxplus(self, x, y):
        """Check-node update，可用 min-sum 近似。"""
        x = np.clip(x, -self.llr_max, self.llr_max)
        y = np.clip(y, -self.llr_max, self.llr_max)
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)。"""
        llr_orig = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = llr_orig[bit_reversal_permutation(self.N)]
        llr_ch = np.clip(llr_ch, -self.llr_max, self.llr_max)
        n = self.N
        n_stages = self.n

        msg_r_in = np.zeros(n, dtype=np.float64)
        msg_r_in[self.frozen_pos] = self.llr_max

        hard_ch = (llr_orig < 0).astype(int)
        num_iters = self.max_iter

        msg_l = [[None] * (n_stages + 1)]
        msg_r = [[None] * (n_stages + 1)]

        for ind_it in range(self.max_iter):
            msg_l.append([None] * (n_stages + 1))
            msg_r.append([None] * (n_stages + 1))

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

                r1_out = self._boxplus(r1_in, l2_in + r2_in)
                r2_out = self._boxplus(r1_in, l1_in) + r2_in

                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
                r_out = np.concatenate([r1_out, r2_out])[ind_inv]
                msg_r[ind_it][ind_s + 1] = r_out

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

                l1_out = self._boxplus(l1_in, l2_in + r2_in)
                l2_out = self._boxplus(r1_in, l1_in) + l2_in

                l_out = np.concatenate([l1_out, l2_out])[ind_inv]
                msg_l[ind_it][ind_s] = l_out

            soft = msg_l[ind_it][0]
            u_hat = np.zeros(n, dtype=int)
            u_hat[self.info_pos] = (soft[self.info_pos] <= 0).astype(int)

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_ch):
                num_iters = ind_it + 1
                break
        else:
            soft = msg_l[self.max_iter - 1][0]
            u_hat = np.zeros(n, dtype=int)
            u_hat[self.info_pos] = (soft[self.info_pos] <= 0).astype(int)

        return u_hat, num_iters


if __name__ == "__main__":
    from construction import ga_construction
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    u = np.zeros(N, dtype=int)
    u[info_idx] = np.random.randint(0, 2, K)
    llr = compute_llr(
        bpsk_modulate(polar_encode(u)), eb_n0_to_sigma(10.0, K / N)
    )
    bp = BPDecoder(N, frozen_bits)
    u_hat, iters = bp.decode(llr)
    print("BP iters:", iters)
    print("Info match:", np.array_equal(u_hat[info_idx], u[info_idx]))
