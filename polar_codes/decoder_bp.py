"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


def _boxplus(x, y, use_min_sum=False, alpha=0.9375):
    """Check-node 更新（boxplus / min-sum）"""
    if use_min_sum:
        return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))
    return f_operation(x, y)


class BPDecoder:
    """BP 译码器，基于极化码因子图（Sionna/Arikan 索引约定）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375, use_min_sum=False):
        self.N = N
        self.n_stages = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.use_min_sum = use_min_sum
        self.llr_max = 1e6
        self._stage_indices = self._precompute_stage_indices()

    def _precompute_stage_indices(self):
        """预计算各 stage 的 (ind_1, ind_2, ind_inv)"""
        n = self.N
        stages = []
        for ind_s in range(self.n_stages):
            ind_range = np.arange(n // 2)
            ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
            ind_2 = ind_1 + 2 ** ind_s
            ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
            stages.append((ind_1, ind_2, ind_inv))
        return stages

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.N
        n_stages = self.n_stages

        msg_r_in = np.zeros(n, dtype=np.float64)
        msg_r_in[self.frozen_bits] = self.llr_max

        msg_l = [None] * (n_stages + 1)
        msg_r = [None] * (n_stages + 1)

        hard_ch = (llr_ch < 0).astype(int)
        num_iters = 0

        for ind_it in range(self.max_iter):
            num_iters = ind_it + 1

            for ind_s, (ind_1, ind_2, ind_inv) in enumerate(self._stage_indices):
                if ind_s == n_stages - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                elif ind_it == 0:
                    l1_in = np.zeros(len(ind_1))
                    l2_in = np.zeros(len(ind_2))
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

                r1_out = _boxplus(r1_in, l2_in + r2_in, self.use_min_sum, self.alpha)
                r2_out = _boxplus(r1_in, l1_in, self.use_min_sum, self.alpha) + r2_in

                r_out = np.empty(n)
                combined_idx = np.concatenate([ind_1, ind_2])
                r_out[combined_idx] = np.concatenate([r1_out, r2_out])
                msg_r[ind_s + 1] = r_out

            for ind_s in range(n_stages - 1, -1, -1):
                ind_1, ind_2, _ = self._stage_indices[ind_s]

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

                l1_out = _boxplus(l1_in, l2_in + r2_in, self.use_min_sum, self.alpha)
                l2_out = _boxplus(r1_in, l1_in, self.use_min_sum, self.alpha) + l2_in

                l_out = np.empty(n)
                combined_idx = np.concatenate([ind_1, ind_2])
                l_out[combined_idx] = np.concatenate([l1_out, l2_out])
                msg_l[ind_s] = l_out

            u_hat = np.zeros(n, dtype=int)
            llr_left = msg_l[0]
            u_hat[self.info_indices] = (llr_left[self.info_indices] < 0).astype(int)

            if np.array_equal(polar_encode(u_hat), hard_ch):
                return u_hat, num_iters

        u_hat = np.zeros(n, dtype=int)
        llr_left = msg_l[0]
        u_hat[self.info_indices] = (llr_left[self.info_indices] < 0).astype(int)
        return u_hat, num_iters
