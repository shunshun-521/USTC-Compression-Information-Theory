"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode


def _boxplus(a, b, alpha):
    """min-sum box-plus；标量/向量化。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    # 对大 LLR 使用 min-sum，对小 LLR 使用精确 box-plus
    out = alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))
    small = (np.abs(a) < 30) & (np.abs(b) < 30)
    if np.any(small):
        ax = np.clip(a, -30, 30)
        bx = np.clip(b, -30, 30)
        exact = np.log1p(np.exp(ax + bx)) - np.log(np.exp(ax) + np.exp(bx))
        out = np.where(small, exact, out)
    return out


class BPDecoder:
    """BP 译码器（Sionna/Arikan 因子图结构）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n_stages = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]
        self.llr_max = 19.3

    def _stage_indices(self, stage):
        ind_range = np.arange(self.N // 2)
        ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** stage)
        ind_2 = ind_1 + 2 ** stage
        ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
        return ind_1, ind_2, ind_inv

    def decode(self, llr_ch):
        llr_ch = np.clip(np.asarray(llr_ch, dtype=np.float64), -self.llr_max, self.llr_max)
        llr_ch = -llr_ch  # 与 Sionna 一致：内部使用 negated LLR

        msg_l = [[None] * (self.n_stages + 1) for _ in range(self.max_iter)]
        msg_r = [[None] * (self.n_stages + 1) for _ in range(self.max_iter)]

        msg_r_in = np.zeros(self.N, dtype=np.float64)
        msg_r_in[self.frozen_idx] = self.llr_max

        num_iters = 0
        u_hat = np.zeros(self.N, dtype=int)

        for ind_it in range(self.max_iter):
            for ind_s in range(self.n_stages):
                ind_1, ind_2, ind_inv = self._stage_indices(ind_s)

                if ind_s == self.n_stages - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                elif ind_it == 0:
                    l1_in = np.zeros(self.N // 2)
                    l2_in = np.zeros(self.N // 2)
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

                r1_out = _boxplus(r1_in, l2_in + r2_in, self.alpha)
                r2_out = _boxplus(r1_in, l1_in, self.alpha) + r2_in
                r_out = np.concatenate([r1_out, r2_out])[ind_inv]
                msg_r[ind_it][ind_s + 1] = r_out

            for ind_s in range(self.n_stages - 1, -1, -1):
                ind_1, ind_2, ind_inv = self._stage_indices(ind_s)

                if ind_s == self.n_stages - 1:
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

                l1_out = _boxplus(l1_in, l2_in + r2_in, self.alpha)
                l2_out = _boxplus(r1_in, l1_in, self.alpha) + l2_in
                l_out = np.concatenate([l1_out, l2_out])[ind_inv]
                msg_l[ind_it][ind_s] = l_out

            soft = msg_l[ind_it][0]
            u_hat_full = (soft <= 0).astype(int)
            u_hat_full[self.frozen_idx] = 0
            num_iters = ind_it + 1

            x_hat = polar_encode(u_hat_full)
            hard_x = (llr_ch >= 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                return u_hat_full, num_iters

        soft = msg_l[num_iters - 1][0]
        u_hat_full = (soft <= 0).astype(int)
        u_hat_full[self.frozen_idx] = 0
        return u_hat_full, num_iters
