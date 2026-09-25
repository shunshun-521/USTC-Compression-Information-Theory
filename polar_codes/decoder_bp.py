"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from decoder_sc import f_operation, reorder_channel_llr
from encoder import polar_encode


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    列 0：信源比特端；列 n：信道接收端。
    """

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        """
        参数：
            N: 码长
            frozen_bits: 长度 N 的 bool 数组
            max_iter: 最大迭代次数
            alpha: min-sum 修正因子（典型值 0.9375）
        """
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def _f_ms(self, a, b):
        """min-sum f 运算，带 alpha 修正"""
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """
        主译码函数。

        参数：
            llr_ch: 长度 N 的信道接收 LLR（对应因子图最右列）

        返回：
            u_hat: 长度 N 的估计源序列
            num_iters: 实际迭代次数
        """
        N = self.N
        n = self.n
        llr_ch = reorder_channel_llr(llr_ch)

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(self.max_iter):
            num_iters = it + 1

            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx1 = i + k
                        idx2 = i + k + s
                        L[idx1, j - 1] = self._f_ms(
                            R[idx1, j] + L[idx2, j],
                            L[idx1, j],
                        )
                        L[idx2, j - 1] = (
                            self._f_ms(R[idx1, j], L[idx1, j]) + L[idx2, j]
                        )

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx1 = i + k
                        idx2 = i + k + s
                        R[idx1, j + 1] = self._f_ms(
                            R[idx2, j] + L[idx2, j + 1],
                            R[idx1, j],
                        )
                        R[idx2, j + 1] = (
                            self._f_ms(R[idx1, j], L[idx1, j + 1]) + R[idx2, j]
                        )

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(np.int8)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, x_hard):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(np.int8)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
