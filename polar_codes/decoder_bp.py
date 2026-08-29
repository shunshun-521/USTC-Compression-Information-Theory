"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation


class BPDecoder:
    """BP 译码器（因子图 min-sum）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6
        self.br = bit_reversal_permutation(N)

    def _f_boxplus(self, x, y):
        return self.alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))

    def decode(self, llr_ch):
        """
        主译码函数。

        返回：
            u_hat: 长度 N 的估计源序列
            num_iters: 实际迭代次数
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        # L[i][j]: 从右到左消息；R[i][j]: 从左到右消息
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch

        frozen_idx = np.where(self.frozen_bits.astype(bool))[0]
        R[frozen_idx, 0] = self.large

        hard_ch = (llr_ch < 0).astype(int)
        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            # 从右到左更新 L
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        Li = i + k
                        Li_s = i + k + s
                        L[Li, j - 1] = self._f_boxplus(
                            R[Li, j] + L[Li_s, j], L[Li, j]
                        )
                        L[Li_s, j - 1] = self._f_boxplus(
                            R[Li, j], L[Li, j]
                        ) + L[Li_s, j]

            # 从左到右更新 R
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        Li = i + k
                        Li_s = i + k + s
                        R[Li, j + 1] = self._f_boxplus(
                            R[Li_s, j] + L[Li_s, j + 1], R[Li, j]
                        )
                        R[Li_s, j + 1] = self._f_boxplus(
                            R[Li, j], L[Li, j + 1]
                        ) + R[Li_s, j]

            u_hat = self._hard_decision(L, R)
            u_hat[self.frozen_bits.astype(bool)] = 0

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
        else:
            u_hat = self._hard_decision(L, R)
            u_hat[self.frozen_bits.astype(bool)] = 0

        return u_hat.astype(int), num_iters

    def _hard_decision(self, L, R):
        total = L[:, 0] + R[:, 0]
        return (total < 0).astype(int)
