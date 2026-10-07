"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（因子图列 0..n）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N
        br = bit_reversal_permutation(N)
        ch = llr_ch[br]

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self._large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=np.int32)

        for it in range(1, self.max_iter + 1):
            # 右到左更新 L
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx_u = i + k
                        idx_l = i + k + step
                        L[idx_u, j - 1] = self._f_ms(
                            R[idx_u, j] + L[idx_l, j], L[idx_u, j]
                        )
                        L[idx_l, j - 1] = self._f_ms(R[idx_u, j], L[idx_u, j]) + L[idx_l, j]

            # 左到右更新 R
            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx_u = i + k
                        idx_l = i + k + step
                        R[idx_u, j + 1] = self._f_ms(
                            R[idx_l, j] + L[idx_l, j + 1], R[idx_u, j]
                        )
                        R[idx_l, j + 1] = self._f_ms(R[idx_u, j], L[idx_u, j + 1]) + R[idx_l, j]

            total = L[:, 0] + R[:, 0]
            u_hat = np.where(total >= 0, 0, 1).astype(np.int32)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, (llr_ch < 0).astype(np.int32)):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.where(total >= 0, 0, 1).astype(np.int32)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
