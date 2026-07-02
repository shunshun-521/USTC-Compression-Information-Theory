"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode, bit_reversal_permutation


def _f_minsum(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器（参考 Sionna/Arikan 因子图结构）。"""

    LLR_MAX = 19.3

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_pos = np.where(self.frozen_bits)[0]
        self.info_pos = np.where(~self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self._rev = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        """
        主译码函数。
        llr_ch: 信道 LLR（自然顺序，对应发送码字 x）
        """
        llr_natural = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = llr_natural[self._rev]
        N, n = self.N, self.n

        msg_r_in = np.zeros(N, dtype=np.float64)
        msg_r_in[self.frozen_pos] = self.LLR_MAX

        msg_l = [None] * (n + 1)
        msg_r = [None] * (n + 1)

        num_iters = self.max_iter

        for it in range(self.max_iter):
            # 左到右更新 R 消息
            for s in range(n):
                ind_range = np.arange(N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** s)
                ind_2 = ind_1 + 2 ** s

                if s == n - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                elif it == 0:
                    l1_in = np.zeros(N // 2)
                    l2_in = np.zeros(N // 2)
                else:
                    l_prev = msg_l[s + 1]
                    l1_in = l_prev[ind_1]
                    l2_in = l_prev[ind_2]

                if s == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_prev = msg_r[s]
                    r1_in = r_prev[ind_1]
                    r2_in = r_prev[ind_2]

                r1_out = _f_minsum(r1_in, l2_in + r2_in, self.alpha)
                r2_out = _f_minsum(r1_in, l1_in, self.alpha) + r2_in

                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
                msg_r[s + 1] = np.concatenate([r1_out, r2_out])[ind_inv]

            # 右到左更新 L 消息
            for s in range(n - 1, -1, -1):
                ind_range = np.arange(N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** s)
                ind_2 = ind_1 + 2 ** s
                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))

                if s == n - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                else:
                    l_prev = msg_l[s + 1]
                    l1_in = l_prev[ind_1]
                    l2_in = l_prev[ind_2]

                if s == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_prev = msg_r[s]
                    r1_in = r_prev[ind_1]
                    r2_in = r_prev[ind_2]

                l1_out = _f_minsum(l1_in, l2_in + r2_in, self.alpha)
                l2_out = _f_minsum(r1_in, l1_in, self.alpha) + l2_in

                msg_l[s] = np.concatenate([l1_out, l2_out])[ind_inv]

            u_hat = self._decide(msg_l[0])
            if self._early_stop(u_hat, llr_natural):
                num_iters = it + 1
                break
        else:
            u_hat = self._decide(msg_l[0])

        return u_hat.astype(int), num_iters

    def _decide(self, llr_left):
        """从第 0 层 L 消息判决全码字。"""
        u_hat = np.zeros(self.N, dtype=int)
        total = llr_left
        u_hat[self.info_pos] = (total[self.info_pos] < 0).astype(int)
        return u_hat

    def _early_stop(self, u_hat, llr_natural):
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_natural < 0).astype(int)
        return np.array_equal(x_hat, hard_ch)
