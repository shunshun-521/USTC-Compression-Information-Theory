"""
极化码 BP（置信传播）译码器
基于因子图 min-sum 近似，含早停
"""
import numpy as np

from encoder import polar_encode


def _boxplus_ms(x, y, alpha):
    """min-sum 近似的 boxplus（check-node 更新）"""
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    LLR_MAX = 19.3

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n_stages = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha

    def _stage_indices(self, stage):
        """返回当前 stage 的 (ind_1, ind_2) 索引对"""
        ind_range = np.arange(self.N // 2)
        ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** stage)
        ind_2 = ind_1 + 2 ** stage
        return ind_1, ind_2

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.N
        n_stages = self.n_stages
        alpha = self.alpha

        msg_r_in = np.zeros(n)
        msg_r_in[self.frozen_idx] = self.LLR_MAX

        msg_l = [None] * (n_stages + 1)
        msg_r = [None] * (n_stages + 1)

        num_iters = 0
        for ind_it in range(self.max_iter):
            num_iters = ind_it + 1

            # 左到右更新 R
            for ind_s in range(n_stages):
                ind_1, ind_2 = self._stage_indices(ind_s)

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

                r1_out = _boxplus_ms(r1_in, l2_in + r2_in, alpha)
                r2_out = _boxplus_ms(r1_in, l1_in, alpha) + r2_in

                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
                r_out = np.concatenate([r1_out, r2_out])[ind_inv]
                msg_r[ind_s + 1] = r_out

            # 右到左更新 L
            for ind_s in range(n_stages - 1, -1, -1):
                ind_1, ind_2 = self._stage_indices(ind_s)

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

                l1_out = _boxplus_ms(l1_in, l2_in + r2_in, alpha)
                l2_out = _boxplus_ms(r1_in, l1_in, alpha) + l2_in

                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
                l_out = np.concatenate([l1_out, l2_out])[ind_inv]
                msg_l[ind_s] = l_out

            u_hat = (msg_l[0] < 0).astype(int)
            u_hat[self.frozen_bits] = 0
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(polar_encode(u_hat), x_hard):
                break

        u_hat = (msg_l[0] < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
