"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import _map_channel_llrs, f_operation


class BPDecoder:
    """
    BP 译码器（因子图 n+1 列，列 0 为信源端，列 n 为信道端）。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.LARGE = 1e6

    def _f_minsum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat，实际迭代次数
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_nat = _map_channel_llrs(llr_ch, self.N)
        n = self.n
        N = self.N

        # L[i,j]: 从右向左消息；R[i,j]: 从左向右消息
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_nat
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            # 从右到左更新 L（列 n 到 1）
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    L[i, j - 1] = self._f_minsum(
                        R[i, j] + L[i + s, j], L[i, j]
                    )
                    L[i + s, j - 1] = (
                        self._f_minsum(R[i, j], L[i, j]) + L[i + s, j]
                    )

            # 从左到右更新 R（列 0 到 n-1）
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    R[i, j + 1] = self._f_minsum(
                        R[i + s, j] + L[i + s, j + 1], R[i, j]
                    )
                    R[i + s, j + 1] = (
                        self._f_minsum(R[i, j], L[i, j + 1]) + R[i + s, j]
                    )

            # 早停检查
            total = L[:, 0] + R[:, 0]
            u_hat = np.where(total >= 0, 0, 1).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            br = bit_reversal_permutation(N)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.where(total >= 0, 0, 1).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
