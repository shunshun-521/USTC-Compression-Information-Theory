"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
参考 Arikan BP 因子图与 Sionna PolarBPDecoder 的消息传递结构
"""
import math

import numpy as np

from encoder import polar_encode


def _f_minsum(a, b, alpha):
    """min-sum 近似 f 运算（boxplus 近似）。"""
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """
    BP 译码器。

    因子图有 n+1 层（0 到 n）。输入 LLR 须与 SC 一致（使用 channel.channel_llr
    的比特倒序对齐），冻结位在左侧 R 消息初始化为大正数（倾向 u=0）。
    """

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def _stage_indices(self, stage):
        """返回当前 stage 的 (ind_1, ind_2) 索引对及合并逆置换。"""
        ind_range = np.arange(self.N // 2)
        ind_1 = (ind_range * 2 - np.mod(ind_range, 2**stage)).astype(int)
        ind_2 = ind_1 + (1 << stage)
        ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
        return ind_1, ind_2, ind_inv

    def decode(self, llr_ch):
        """
        主译码函数。

        参数：
            llr_ch: 长度 N 的自然序信道 LLR（LLR>0 倾向比特 0）

        返回：
            u_hat: 长度 N 的估计源序列
            num_iters: 实际迭代次数
        """
        N, n = self.N, self.n
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        msg_r_in = np.zeros(N, dtype=np.float64)
        msg_r_in[self.frozen_bits] = self.LARGE

        msg_l = [[None] * (n + 1) for _ in range(self.max_iter)]
        msg_r = [[None] * (n + 1) for _ in range(self.max_iter)]

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(self.max_iter):
            num_iters = it + 1

            # 从左到右更新 R 消息
            for s in range(n):
                ind_1, ind_2, ind_inv = self._stage_indices(s)

                if s == n - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                elif it == 0:
                    l1_in = np.zeros(N // 2)
                    l2_in = np.zeros(N // 2)
                else:
                    l_prev = msg_l[it - 1][s + 1]
                    l1_in = l_prev[ind_1]
                    l2_in = l_prev[ind_2]

                if s == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_prev = msg_r[it][s]
                    r1_in = r_prev[ind_1]
                    r2_in = r_prev[ind_2]

                r1_out = _f_minsum(r1_in, l2_in + r2_in, self.alpha)
                r2_out = _f_minsum(r1_in, l1_in, self.alpha) + r2_in
                msg_r[it][s + 1] = np.concatenate([r1_out, r2_out])[ind_inv]

            # 从右到左更新 L 消息
            for s in range(n - 1, -1, -1):
                ind_1, ind_2, ind_inv = self._stage_indices(s)

                if s == n - 1:
                    l1_in = llr_ch[ind_1]
                    l2_in = llr_ch[ind_2]
                else:
                    l_prev = msg_l[it][s + 1]
                    l1_in = l_prev[ind_1]
                    l2_in = l_prev[ind_2]

                if s == 0:
                    r1_in = msg_r_in[ind_1]
                    r2_in = msg_r_in[ind_2]
                else:
                    r_prev = msg_r[it][s]
                    r1_in = r_prev[ind_1]
                    r2_in = r_prev[ind_2]

                l1_out = _f_minsum(l1_in, l2_in + r2_in, self.alpha)
                l2_out = _f_minsum(r1_in, l1_in, self.alpha) + l2_in
                msg_l[it][s] = np.concatenate([l1_out, l2_out])[ind_inv]

            left_llr = msg_l[it][0]
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if left_llr[i] >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        left_llr = msg_l[num_iters - 1][0]
        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if left_llr[i] >= 0 else 1

        return u_hat, num_iters
