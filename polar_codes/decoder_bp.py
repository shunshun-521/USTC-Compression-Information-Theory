"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math

from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e30

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]

    def _minsum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.br]
        n, N = self.n, self.N

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)

        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_idx] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for s in range(n - 1, -1, -1):
                step = 1 << s
                for j in range(0, N, 2 * step):
                    for k in range(step):
                        idx = j + k
                        idx2 = idx + step
                        L[s, idx] = self._minsum(
                            R[s, idx] + L[s + 1, idx2],
                            L[s + 1, idx],
                        )
                        L[s, idx2] = self._minsum(R[s, idx], L[s + 1, idx]) + L[s + 1, idx2]

            for s in range(n):
                step = 1 << s
                for j in range(0, N, 2 * step):
                    for k in range(step):
                        idx = j + k
                        idx2 = idx + step
                        R[s + 1, idx] = self._minsum(
                            R[s, idx2] + L[s + 1, idx2],
                            R[s, idx],
                        )
                        R[s + 1, idx2] = self._minsum(R[s, idx], L[s + 1, idx2]) + R[s, idx2]

            total = L[0, :] + R[0, :]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat[self.br], hard_ch):
                break

        total = L[0, :] + R[0, :]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
