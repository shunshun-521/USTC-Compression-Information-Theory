"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from decoder_sc import f_operation, _prepare_llr
from encoder import polar_encode


class BPDecoder:
    """BP 译码器，因子图 min-sum 近似 + 早停。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6

    def _minsum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr = _prepare_llr(llr_ch)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, 0] = llr
        R[:, n] = 0.0
        R[self.frozen_bits, 0] = self._large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        idx = i + t
                        L[idx, j] = self._minsum(
                            R[idx, j - 1] + L[idx + s, j - 1],
                            L[idx, j - 1],
                        )
                        L[idx + s, j] = self._minsum(
                            R[idx, j - 1],
                            L[idx, j - 1],
                        ) + L[idx + s, j - 1]

            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        idx = i + t
                        R[idx, j] = self._minsum(
                            R[idx + s, j + 1] + L[idx + s, j + 1],
                            R[idx, j + 1],
                        )
                        R[idx + s, j] = self._minsum(
                            R[idx, j + 1],
                            L[idx, j + 1],
                        ) + R[idx + s, j + 1]

            num_iters = it
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, n] + R[i, n]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            llr_perm = _prepare_llr(llr_ch)
            hard_ch = (llr_perm < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, n] + R[i, n]) >= 0 else 1

        return u_hat, num_iters
