"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation, _frozen_mask_to_info_pos


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        self.frozen_indices = np.where(
            self.frozen_bits.astype(bool)
            if self.frozen_bits.dtype == bool
            else self.frozen_bits != 0
        )[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        rev = bit_reversal_permutation(N)
        ch = llr_ch[rev]

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n] = ch
        R[0] = 0.0
        R[0, self.frozen_indices] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # 右到左更新 L
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    L[j - 1, i] = self._f_min_sum(
                        R[j, i] + L[j, i + s], L[j, i]
                    )
                    L[j - 1, i + s] = self._f_min_sum(
                        R[j, i], L[j, i]
                    ) + L[j, i + s]

            # 左到右更新 R
            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    R[j, i] = self._f_min_sum(
                        R[j, i + s] + L[j, i + s], R[j - 1, i]
                    )
                    R[j, i + s] = self._f_min_sum(
                        R[j - 1, i], L[j, i]
                    ) + R[j, i + s]

            num_iters = it

            # 早停检查
            total = L[0] + R[0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_indices] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (ch < 0).astype(int)
            x_hat_br = x_hat[rev]
            if np.array_equal(x_hat_br, (llr_ch < 0).astype(int)):
                break

        total = L[0] + R[0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_indices] = 0
        return u_hat, num_iters
