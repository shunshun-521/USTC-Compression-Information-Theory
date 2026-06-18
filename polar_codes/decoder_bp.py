"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import bit_reversal_permutation, polar_encode
from channel import hard_decision_llr


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    列 0：信源比特端；列 n：信道接收端。

    消息：
      L[i][j]: 第 i 行、第 j 列的 L 消息（从右到左）
      R[i][j]: 第 i 行、第 j 列的 R 消息（从左到右）
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self.large = 1e6

    def _f_min_sum(self, a, b):
        return self.alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

    def decode(self, llr_ch):
        """
        主译码函数。

        参数：
            llr_ch: 长度 N 的信道接收 LLR（对应因子图最右列）

        返回：
            u_hat: 长度 N 的估计源序列
            num_iters: 实际迭代次数
        """
        N, n = self.N, self.n
        llr = np.asarray(llr_ch, dtype=np.float64)[self.br]

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr

        frozen_idx = np.where(self.frozen_bits)[0]
        R[frozen_idx, 0] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # 从右到左更新 L
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        idx = i + t
                        idx2 = idx + s
                        L[idx, j - 1] = self._f_min_sum(
                            R[idx, j] + L[idx2, j],
                            L[idx, j],
                        )
                        L[idx2, j - 1] = self._f_min_sum(
                            R[idx, j],
                            L[idx, j],
                        ) + L[idx2, j]

            # 从左到右更新 R
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        idx = i + t
                        idx2 = idx + s
                        R[idx, j + 1] = self._f_min_sum(
                            R[idx2, j] + L[idx2, j + 1],
                            R[idx, j],
                        )
                        R[idx2, j + 1] = self._f_min_sum(
                            R[idx, j],
                            L[idx, j + 1],
                        ) + R[idx2, j]

            # 判决与早停
            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break
            num_iters = it

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
