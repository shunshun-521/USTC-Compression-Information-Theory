"""
极化码 BP（置信传播）译码器
基于 Sionna/Arikan 因子图结构，min-sum 近似，含早停
"""
import math
import numpy as np

from encoder import polar_encode


def _minsum_boxplus(x, y, alpha):
    x = np.clip(x, -20.0, 20.0)
    y = np.clip(y, -20.0, 20.0)
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.llr_max = 19.3

    def _stage_indices(self, stage):
        ind_range = np.arange(self.N // 2)
        ind_1 = ind_range * 2 - np.mod(ind_range, 2**stage)
        ind_2 = ind_1 + 2**stage
        ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
        return ind_1, ind_2, ind_inv

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        alpha = self.alpha

        msg_r_in = np.zeros(N, dtype=np.float64)
        msg_r_in[self.frozen_idx] = self.llr_max

        msg_l = [None] * (n + 1)
        msg_r = [None] * (n + 1)

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

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
                    l_in = msg_l[stage + 1]
                    l1_in = l_in[ind_1]
                    l2_in = l_in[ind_2]

                if stage == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_in = msg_r[stage]
                    r1_in = r_in[ind_1]
                    r2_in = r_in[ind_2]

                r1_out = _minsum_boxplus(r1_in, l2_in + r2_in, alpha)
                r2_out = _minsum_boxplus(r1_in, l1_in, alpha) + r2_in
                r_out = np.concatenate([r1_out, r2_out])[ind_inv]
                msg_r[stage + 1] = r_out

            for stage in range(n - 1, -1, -1):
                ind_1, ind_2, ind_inv = self._stage_indices(stage)

                if stage == n - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                else:
                    l_in = msg_l[stage + 1]
                    l1_in = l_in[ind_1]
                    l2_in = l_in[ind_2]

                if stage == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_in = msg_r[stage]
                    r1_in = r_in[ind_1]
                    r2_in = r_in[ind_2]

                l1_out = _minsum_boxplus(l1_in, l2_in + r2_in, alpha)
                l2_out = _minsum_boxplus(r1_in, l1_in, alpha) + l2_in
                l_out = np.concatenate([l1_out, l2_out])[ind_inv]
                msg_l[stage] = l_out

            num_iters = it + 1
            soft = msg_l[0]
            for i in range(N):
                u_hat[i] = 0 if soft[i] > 0 else 1
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        soft = msg_l[0]
        for i in range(N):
            u_hat[i] = 0 if soft[i] > 0 else 1
        u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
