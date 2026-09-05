"""
极化码 BP（置信传播）译码器
基于因子图 stage 结构（参考 Sionna PolarBPDecoder），含早停机制。
"""
import math
import numpy as np

from encoder import polar_encode


def _boxplus(x, y, llr_max=19.3):
    """Check-node update (exact box-plus, 与 Sionna 一致)."""
    x = np.clip(x, -llr_max, llr_max)
    y = np.clip(y, -llr_max, llr_max)
    return np.log(1.0 + np.exp(x + y)) - np.log(np.exp(x) + np.exp(y))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375, llr_max=19.3):
        self.N = N
        self.n = int(math.log2(N))
        self.max_iter = max_iter
        self.alpha = alpha
        self.llr_max = llr_max
        fb = np.asarray(frozen_bits)
        self.frozen_mask = fb.astype(bool) if fb.dtype == bool else (fb != 0)
        self.frozen_idx = np.where(self.frozen_mask)[0]
        self.info_idx = np.where(~self.frozen_mask)[0]

    def decode(self, llr_ch):
        # compute_llr 输出已与 Sionna _decode_bp 内部 LLR 约定一致（正 -> 比特 0）
        llr_ext = np.asarray(llr_ch, dtype=np.float64).reshape(1, -1)
        bs = 1
        n_stages = self.n
        N = self.N

        msg_l = [[None] * (n_stages + 1) for _ in range(self.max_iter)]
        msg_r = [[None] * (n_stages + 1) for _ in range(self.max_iter)]
        msg_r_in = np.zeros((bs, N), dtype=np.float64)
        msg_r_in[:, self.frozen_idx] = self.llr_max

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for ind_it in range(self.max_iter):
            num_iters = ind_it + 1

            for ind_s in range(n_stages):
                ind_range = np.arange(N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
                ind_2 = ind_1 + 2 ** ind_s

                if ind_s == n_stages - 1:
                    l1_in = llr_ext[:, ind_1]
                    l2_in = llr_ext[:, ind_2]
                elif ind_it == 0:
                    l1_in = np.zeros((bs, N // 2))
                    l2_in = np.zeros((bs, N // 2))
                else:
                    l_in = msg_l[ind_it - 1][ind_s + 1]
                    l1_in = l_in[:, ind_1]
                    l2_in = l_in[:, ind_2]

                if ind_s == 0:
                    r1_in = msg_r_in[:, ind_1]
                    r2_in = msg_r_in[:, ind_2]
                else:
                    r_in = msg_r[ind_it][ind_s]
                    r1_in = r_in[:, ind_1]
                    r2_in = r_in[:, ind_2]

                r1_out = _boxplus(r1_in, l2_in + r2_in, self.llr_max)
                r2_out = _boxplus(r1_in, l1_in, self.llr_max) + r2_in
                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
                msg_r[ind_it][ind_s + 1] = np.concatenate([r1_out, r2_out], axis=1)[:, ind_inv]

            for ind_s in range(n_stages - 1, -1, -1):
                ind_range = np.arange(N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
                ind_2 = ind_1 + 2 ** ind_s
                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))

                if ind_s == n_stages - 1:
                    l1_in = llr_ext[:, ind_1]
                    l2_in = llr_ext[:, ind_2]
                else:
                    l_in = msg_l[ind_it][ind_s + 1]
                    l1_in = l_in[:, ind_1]
                    l2_in = l_in[:, ind_2]

                if ind_s == 0:
                    r1_in = msg_r_in[:, ind_1]
                    r2_in = msg_r_in[:, ind_2]
                else:
                    r_in = msg_r[ind_it][ind_s]
                    r1_in = r_in[:, ind_1]
                    r2_in = r_in[:, ind_2]

                l1_out = _boxplus(l1_in, l2_in + r2_in, self.llr_max)
                l2_out = _boxplus(r1_in, l1_in, self.llr_max) + l2_in
                msg_l[ind_it][ind_s] = np.concatenate([l1_out, l2_out], axis=1)[:, ind_inv]

            l0 = msg_l[ind_it][0][0]
            for i in range(N):
                if self.frozen_mask[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if l0[i] > 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ext[0] < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        l0 = msg_l[num_iters - 1][0][0]
        for i in range(N):
            if self.frozen_mask[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if l0[i] > 0 else 1

        return u_hat, num_iters
