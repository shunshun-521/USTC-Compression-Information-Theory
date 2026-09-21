"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode


def _boxplus_minsum(x, y, alpha):
    """Min-sum 近似 boxplus。"""
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        if 2 ** self.n != N:
            raise ValueError("N must be a power of 2")
        frozen_bits = np.asarray(frozen_bits)
        if frozen_bits.dtype == bool:
            self.frozen = np.where(frozen_bits)[0]
        else:
            self.frozen = np.where(frozen_bits.astype(int) == 1)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.llr_max = 1e6

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：
            u_hat: 长度 N 的估计源序列
            num_iters: 实际迭代次数
        """
        llr_natural = np.asarray(llr_ch, dtype=np.float64)
        llr_dec = llr_natural.copy()

        n = self.n
        N = self.N
        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        msg_r_in = np.zeros(N, dtype=np.float64)
        msg_r_in[self.frozen] = self.llr_max

        msg_l_prev = None

        for it in range(self.max_iter):
            msg_l = [None] * (n + 1)
            msg_r = [None] * (n + 1)

            for stage in range(n):
                half = N // 2
                ind_range = np.arange(half)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** stage)
                ind_2 = ind_1 + 2 ** stage
                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))

                if stage == n - 1:
                    l1_in = llr_dec[ind_1]
                    l2_in = llr_dec[ind_2]
                elif it == 0:
                    l1_in = np.zeros(half, dtype=np.float64)
                    l2_in = np.zeros(half, dtype=np.float64)
                else:
                    l_in = msg_l_prev[stage + 1]
                    l1_in = l_in[ind_1]
                    l2_in = l_in[ind_2]

                if stage == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_in = msg_r[stage]
                    r1_in = r_in[ind_1]
                    r2_in = r_in[ind_2]

                r1_out = _boxplus_minsum(r1_in, l2_in + r2_in, self.alpha)
                r2_out = _boxplus_minsum(r1_in, l1_in, self.alpha) + r2_in
                r_concat = np.concatenate([r1_out, r2_out])
                r_out = np.empty(N, dtype=np.float64)
                r_out[ind_inv] = r_concat
                msg_r[stage + 1] = r_out

            for stage in range(n - 1, -1, -1):
                half = N // 2
                ind_range = np.arange(half)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** stage)
                ind_2 = ind_1 + 2 ** stage
                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))

                if stage == n - 1:
                    l1_in = llr_dec[ind_1]
                    l2_in = llr_dec[ind_2]
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

                l1_out = _boxplus_minsum(l1_in, l2_in + r2_in, self.alpha)
                l2_out = _boxplus_minsum(r1_in, l1_in, self.alpha) + l2_in
                l_concat = np.concatenate([l1_out, l2_out])
                l_out = np.empty(N, dtype=np.float64)
                l_out[ind_inv] = l_concat
                msg_l[stage] = l_out

            msg_l_prev = msg_l

            for i in range(N):
                total = msg_l[0][i] + msg_r_in[i]
                u_hat[i] = 0 if total >= 0 else 1
            u_hat[self.frozen] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_natural < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it + 1
                break

        return u_hat.astype(int), num_iters
