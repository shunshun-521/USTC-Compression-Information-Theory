"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math
from encoder import polar_encode, bit_reversed
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)。"""
        N, n = self.N, self.n
        LARGE = 1e6

        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))

        L[:, 0] = llr_ch.copy()
        R[:, n] = 0.0
        R[self.frozen_idx, n] = LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for s in range(n):
                block_size = 2 ** (s + 1)
                branch_size = block_size // 2
                for j in range(0, N, block_size):
                    for k in range(branch_size):
                        idx = j + k
                        L[idx, s + 1] = self._f_min_sum(
                            R[idx, s] + L[idx + branch_size, s],
                            L[idx, s]
                        )
                        L[idx + branch_size, s + 1] = self._f_min_sum(
                            R[idx, s], L[idx, s]
                        ) + L[idx + branch_size, s]

            for s in range(n - 1, -1, -1):
                block_size = 2 ** (s + 1)
                branch_size = block_size // 2
                for j in range(0, N, block_size):
                    for k in range(branch_size):
                        idx = j + k
                        R[idx, s] = self._f_min_sum(
                            R[idx + branch_size, s + 1] + L[idx + branch_size, s + 1],
                            R[idx, s + 1]
                        )
                        R[idx + branch_size, s] = self._f_min_sum(
                            R[idx, s + 1], L[idx, s + 1]
                        ) + R[idx + branch_size, s + 1]

            for i in range(N):
                total = L[i, n] + R[i, n]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        for i in range(N):
            total = L[i, n] + R[i, n]
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
