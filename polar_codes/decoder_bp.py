"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode_core, bit_reversal_permutation
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.rev = bit_reversal_permutation(N)
        self.large = 1e6

    def _f_ms(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        # L[i,j]: 从右到左；R[i,j]: 从左到右
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        # 信道 LLR 在极化变换域
        L[:, n] = llr_ch[self.rev]
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            # 右到左更新 L
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_u = i + k
                        idx_l = i + k + s
                        L[idx_u, j - 1] = self._f_ms(
                            R[idx_u, j] + L[idx_l, j], L[idx_u, j]
                        )
                        L[idx_l, j - 1] = self._f_ms(
                            R[idx_u, j], L[idx_u, j]
                        ) + L[idx_l, j]

            # 左到右更新 R
            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_u = i + k
                        idx_l = i + k + s
                        R[idx_u, j] = self._f_ms(
                            R[idx_l, j] + L[idx_l, j], R[idx_u, j - 1]
                        )
                        R[idx_l, j] = self._f_ms(
                            R[idx_u, j - 1], L[idx_u, j]
                        ) + R[idx_l, j - 1]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(np.int8)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode_core(u_hat)
            x_hat_tx = x_hat[self.rev]
            hard_ch = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat_tx, hard_ch):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(np.int8)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
