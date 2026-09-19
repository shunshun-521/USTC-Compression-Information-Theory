"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation, _reorder_channel_llrs


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.inv_br = np.argsort(bit_reversal_permutation(N))

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """
        主译码函数。

        返回：
            u_hat: 长度 N 的估计源序列
            num_iters: 实际迭代次数
        """
        llr_std = _reorder_channel_llrs(np.asarray(llr_ch, dtype=np.float64))
        n = self.n
        N = self.N
        large = 1e6

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_std
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = large

        u_hat = np.zeros(N, dtype=int)
        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            # 从右到左更新 L
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        top = i + k
                        bot = i + k + s
                        L[top, j - 1] = self._f_min_sum(
                            R[top, j - 1] + L[bot, j], L[top, j]
                        )
                        L[bot, j - 1] = self._f_min_sum(
                            R[top, j - 1], L[top, j]
                        ) + L[bot, j]

            # 从左到右更新 R
            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        top = i + k
                        bot = i + k + s
                        R[top, j] = self._f_min_sum(
                            R[bot, j - 1] + L[bot, j], R[top, j - 1]
                        )
                        R[bot, j] = self._f_min_sum(
                            R[top, j - 1], L[top, j]
                        ) + R[bot, j - 1]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
