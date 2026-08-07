"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math
from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """
    BP 译码器。
    """

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

        self.frozen_idx = np.where(self.frozen_bits)[0]

    def _f_min_sum(self, a, b):
        """带修正因子的 min-sum f 运算。"""
        return self.alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        from encoder import bit_reversal_permutation
        rev = bit_reversal_permutation(self.N)
        llr_ch = llr_ch[rev]
        n = self.n
        N = self.N

        # L[i][j]: 从右到左消息; R[i][j]: 从左到右消息
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            # 从右到左更新 L
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        L[idx, j - 1] = self._f_min_sum(
                            R[idx, j] + L[idx + s, j + 1],
                            L[idx, j + 1],
                        )
                        L[idx + s, j - 1] = self._f_min_sum(
                            R[idx, j],
                            L[idx, j + 1],
                        ) + L[idx + s, j + 1]

            # 从左到右更新 R
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        R[idx, j + 1] = self._f_min_sum(
                            R[idx + s, j] + L[idx + s, j + 1],
                            R[idx, j],
                        )
                        R[idx + s, j + 1] = self._f_min_sum(
                            R[idx, j],
                            L[idx, j + 1],
                        ) + R[idx + s, j]

            # 判决与早停
            total_llr = L[:, 0] + R[:, 0]
            u_hat = (total_llr < 0).astype(np.int8)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break
            num_iters = it

        return u_hat, num_iters
