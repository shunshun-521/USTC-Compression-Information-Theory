"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode, _bit_reversal_indices


class BPDecoder:
    """BP 译码器（因子图列 0..n）"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.brp = _bit_reversal_indices(N)

    def _minsum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, num_iters"""
        llr_perm = np.asarray(llr_ch, dtype=np.float64)[self.brp]
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_perm
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, step << 1):
                    for k in range(step):
                        idx = i + k
                        L[idx, j - 1] = self._minsum(
                            R[idx, j] + L[idx + step, j],
                            L[idx, j],
                        )
                        L[idx + step, j - 1] = self._minsum(
                            R[idx, j],
                            L[idx, j],
                        ) + L[idx + step, j]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, step << 1):
                    for k in range(step):
                        idx = i + k
                        R[idx, j + 1] = self._minsum(
                            R[idx + step, j] + L[idx + step, j + 1],
                            R[idx, j],
                        )
                        R[idx + step, j + 1] = self._minsum(
                            R[idx, j],
                            L[idx, j + 1],
                        ) + R[idx + step, j]

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (np.asarray(llr_ch) < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
