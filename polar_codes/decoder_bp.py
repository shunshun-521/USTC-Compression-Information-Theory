"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from scd_core import bit_reversal_permutation
from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _minsum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        # L[i, j]: 从右向左消息，j=0 为信源端，j=n 为信道端
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        br = bit_reversal_permutation(N)
        L[:, n] = llr_ch[br]

        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(self.max_iter):
            num_iters = it + 1

            # 从右到左更新 L（j: n-1 -> 0）
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        idx_s = i + k + s
                        L[idx, j] = self._minsum(
                            R[idx, j + 1] + L[idx_s, j + 1], L[idx, j + 1]
                        )
                        L[idx_s, j] = self._minsum(
                            R[idx, j + 1], L[idx, j + 1]
                        ) + L[idx_s, j + 1]

            # 从左到右更新 R（j: 0 -> n-1）
            for j in range(n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        idx_s = i + k + s
                        R[idx, j + 1] = self._minsum(
                            R[idx_s, j + 1] + L[idx_s, j + 1], R[idx, j]
                        )
                        R[idx_s, j + 1] = self._minsum(
                            R[idx, j], L[idx, j + 1]
                        ) + R[idx_s, j + 1]

            total_llr = L[:, 0] + R[:, 0]
            u_hat = np.where(total_llr >= 0, 0, 1).astype(np.int8)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard_ch):
                break

        total_llr = L[:, 0] + R[:, 0]
        u_hat = np.where(total_llr >= 0, 0, 1).astype(np.int8)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
