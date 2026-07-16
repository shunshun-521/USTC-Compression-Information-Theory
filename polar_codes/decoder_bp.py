"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math
from decoder_sc import f_operation
from encoder import polar_encode, bit_reversal_permutation


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)。"""
        llr_raw = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        rev = bit_reversal_permutation(N)
        llr_perm = llr_raw[rev]

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_perm
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            # 从右到左更新 L 消息（列 n-1 到 0）
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        L[idx, j] = self._f_min_sum(
                            R[idx, j] + L[idx, j + 1],
                            L[idx + s, j + 1],
                        )
                        L[idx + s, j] = (
                            self._f_min_sum(R[idx, j], L[idx, j + 1])
                            + L[idx + s, j + 1]
                        )

            # 从左到右更新 R 消息（列 1 到 n-1）
            for j in range(1, n):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        R[idx, j] = self._f_min_sum(
                            R[idx + s, j] + L[idx + s, j + 1],
                            R[idx, j - 1],
                        )
                        R[idx + s, j] = (
                            self._f_min_sum(R[idx, j - 1], L[idx, j + 1])
                            + R[idx + s, j]
                        )

            for i in range(N):
                u_hat[i] = 0 if self.frozen_bits[i] else (0 if (L[i, 0] + R[i, 0]) >= 0 else 1)

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_raw < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            u_hat[i] = 0 if self.frozen_bits[i] else (0 if (L[i, 0] + R[i, 0]) >= 0 else 1)

        return u_hat, num_iters
