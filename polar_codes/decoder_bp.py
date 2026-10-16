"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import bit_reversal_permutation, polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_ms(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int64)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            # 右到左更新 L
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        idx = i + t
                        L[idx, j - 1] = self._f_ms(
                            R[idx, j] + L[idx + s, j], L[idx, j]
                        )
                        L[idx + s, j - 1] = self._f_ms(R[idx, j], L[idx, j]) + L[idx + s, j]

            # 左到右更新 R
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        idx = i + t
                        R[idx, j + 1] = self._f_ms(
                            R[idx + s, j] + L[idx + s, j + 1], R[idx, j]
                        )
                        R[idx + s, j + 1] = self._f_ms(R[idx, j], L[idx, j + 1]) + R[idx + s, j]

            # 早停
            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(np.int64)
            u_hat[self.frozen_bits] = 0
            x_hat = polar_encode(u_hat)
            br = bit_reversal_permutation(N)
            inv_br = np.empty(N, dtype=np.int64)
            inv_br[br] = np.arange(N)
            hard_ch = (llr_ch[inv_br] < 0).astype(np.int64)
            if np.array_equal(x_hat, hard_ch):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(np.int64)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
