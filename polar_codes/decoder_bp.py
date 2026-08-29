"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode


def boxplus_minsum(a, b, alpha=0.9375):
    """min-sum boxplus with normalization."""
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """
    BP 译码器（flooding schedule，参考 Arikan 因子图结构）。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits > 0)[0]
        self.info_idx = np.where(self.frozen_bits == 0)[0]
        self.llr_max = 1e6

    def _decode_iteration(self, llr_ch, msg_l_prev, msg_r_curr):
        n = self.n
        N = self.N
        alpha = self.alpha

        msg_l = [None] * (n + 1)
        msg_r = [None] * (n + 1)

        for ind_s in range(n):
            ind_range = np.arange(N // 2)
            ind_1 = ind_range * 2 - np.mod(ind_range, 2**ind_s)
            ind_2 = ind_1 + 2**ind_s

            if ind_s == n - 1:
                l1_in = llr_ch[ind_1]
                l2_in = llr_ch[ind_2]
            elif msg_l_prev is None:
                l1_in = np.zeros(N // 2)
                l2_in = np.zeros(N // 2)
            else:
                l_in = msg_l_prev[ind_s + 1]
                l1_in = l_in[ind_1]
                l2_in = l_in[ind_2]

            if ind_s == 0:
                r1_in = msg_r_curr[ind_1]
                r2_in = msg_r_curr[ind_2]
            else:
                r_in = msg_r[ind_s]
                r1_in = r_in[ind_1]
                r2_in = r_in[ind_2]

            r1_out = boxplus_minsum(r1_in, l2_in + r2_in, alpha)
            r2_out = boxplus_minsum(r1_in, l1_in, alpha) + r2_in

            ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
            r_out = np.concatenate([r1_out, r2_out])[ind_inv]
            msg_r[ind_s + 1] = r_out

        for ind_s in range(n - 1, -1, -1):
            ind_range = np.arange(N // 2)
            ind_1 = ind_range * 2 - np.mod(ind_range, 2**ind_s)
            ind_2 = ind_1 + 2**ind_s
            ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))

            if ind_s == n - 1:
                l1_in = llr_ch[ind_1]
                l2_in = llr_ch[ind_2]
            else:
                l_in = msg_l[ind_s + 1]
                l1_in = l_in[ind_1]
                l2_in = l_in[ind_2]

            if ind_s == 0:
                r1_in = msg_r_curr[ind_1]
                r2_in = msg_r_curr[ind_2]
            else:
                r_in = msg_r[ind_s]
                r1_in = r_in[ind_1]
                r2_in = r_in[ind_2]

            l1_out = boxplus_minsum(l1_in, l2_in + r2_in, alpha)
            l2_out = boxplus_minsum(r1_in, l1_in, alpha) + l2_in

            l_out = np.concatenate([l1_out, l2_out])[ind_inv]
            msg_l[ind_s] = l_out

        return msg_l, msg_r

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64).copy()
        n = self.n
        N = self.N

        msg_r_in = np.zeros(N, dtype=np.float64)
        msg_r_in[self.frozen_idx] = self.llr_max

        msg_l_prev = None
        num_iters = self.max_iter

        for it in range(self.max_iter):
            msg_l, msg_r = self._decode_iteration(llr_ch, msg_l_prev, msg_r_in)
            msg_l_prev = msg_l

            total = msg_l[0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it + 1
                break

        total = msg_l_prev[0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
