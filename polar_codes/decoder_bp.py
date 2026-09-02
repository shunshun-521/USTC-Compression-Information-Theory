"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation, _permute_channel_llr


class BPDecoder:
    """BP 译码器（因子图 n+1 列，每列 N 节点）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        n = self.n
        N = self.N
        llr = _permute_channel_llr(llr_ch)

        # L[i, j]: left messages; R[i, j]: right messages; j=0..n
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            # 右到左更新 L
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for block in range(0, N, 2 * s):
                    for i in range(s):
                        idx = block + i
                        L[idx, j - 1] = self._f_ms(
                            R[idx, j] + L[idx + s, j], L[idx, j]
                        )
                        L[idx + s, j - 1] = self._f_ms(
                            R[idx, j], L[idx, j]
                        ) + L[idx + s, j]

            # 左到右更新 R
            for j in range(0, n):
                s = 1 << j
                for block in range(0, N, 2 * s):
                    for i in range(s):
                        idx = block + i
                        R[idx, j + 1] = self._f_ms(
                            R[idx + s, j] + L[idx + s, j + 1], R[idx, j]
                        )
                        R[idx + s, j + 1] = self._f_ms(
                            R[idx, j], L[idx, j + 1]
                        ) + R[idx + s, j]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_x = (llr < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
