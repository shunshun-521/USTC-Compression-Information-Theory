"""
极化码 BP（置信传播）译码器
参考 Sionna / Elkelesh PE 更新，exact box-plus，含早停
"""
import math
import numpy as np

from encoder import polar_encode


def _boxplus(x, y, llr_max=19.3):
    """exact box-plus in LLR domain"""
    x = np.clip(x, -llr_max, llr_max)
    y = np.clip(y, -llr_max, llr_max)
    # 数值稳定：使用 log1p
    t1 = x + y
    ex = np.exp(x)
    ey = np.exp(y)
    et = np.exp(t1)
    # log((1+et)/(ex+ey)) = log1p(et) - log(ex+ey)
    denom = np.log(ex + ey)
    return np.log1p(et) - denom


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_pos = np.where(self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha  # 保留参数兼容；内部使用 exact box-plus
        self._llr_max = 19.3

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n_stages = self.N, self.n

        llr = llr_ch.copy()

        msg_r_in = np.zeros(N, dtype=np.float64)
        msg_r_in[self.frozen_pos] = self._llr_max

        msg_l_iters = []
        msg_r_iters = []

        num_iters = 0
        for ind_it in range(self.max_iter):
            num_iters = ind_it + 1
            msg_l = [None] * (n_stages + 1)
            msg_r = [None] * (n_stages + 1)

            # R：从左到右
            for ind_s in range(n_stages):
                ind_range = np.arange(N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
                ind_2 = ind_1 + 2 ** ind_s

                if ind_s == n_stages - 1:
                    l1_in = llr[ind_1]
                    l2_in = llr[ind_2]
                elif ind_it == 0:
                    l1_in = np.zeros(N // 2)
                    l2_in = np.zeros(N // 2)
                else:
                    l_in = msg_l_iters[ind_it - 1][ind_s + 1]
                    l1_in = l_in[ind_1]
                    l2_in = l_in[ind_2]

                if ind_s == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_in = msg_r[ind_s]
                    r1_in = r_in[ind_1]
                    r2_in = r_in[ind_2]

                r1_out = _boxplus(r1_in, l2_in + r2_in, self._llr_max)
                r2_out = _boxplus(r1_in, l1_in) + r2_in

                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
                r_out = np.concatenate([r1_out, r2_out])
                r_out = r_out[ind_inv]
                msg_r[ind_s + 1] = r_out

            # L：从右到左
            for ind_s in range(n_stages - 1, -1, -1):
                ind_range = np.arange(N // 2)
                ind_1 = ind_range * 2 - np.mod(ind_range, 2 ** ind_s)
                ind_2 = ind_1 + 2 ** ind_s
                ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))

                if ind_s == n_stages - 1:
                    l1_in = llr[ind_1]
                    l2_in = llr[ind_2]
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

                l1_out = _boxplus(l1_in, l2_in + r2_in, self._llr_max)
                l2_out = _boxplus(r1_in, l1_in) + l2_in

                l_out = np.concatenate([l1_out, l2_out])
                l_out = l_out[ind_inv]
                msg_l[ind_s] = l_out

            msg_l_iters.append(msg_l)
            msg_r_iters.append(msg_r)

            # 早停：检查码字一致性
            total = msg_l[0]
            u_hat = np.where(total >= 0, 0, 1).astype(np.int8)
            u_hat[self.frozen_bits] = 0
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard_ch):
                break

        total = msg_l_iters[-1][0]
        u_hat = np.where(total > 0, 0, 1).astype(np.int8)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
