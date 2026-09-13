"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import _prepare_llr, f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器（min-sum + 早停）"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = _prepare_llr(llr_ch)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        for it in range(self.max_iter):
            num_iters = it + 1

            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    L[i:i + s, j] = self._f_min_sum(
                        R[i:i + s, j + 1] + L[i + s:i + 2 * s, j + 1],
                        L[i:i + s, j + 1],
                    )
                    L[i + s:i + 2 * s, j] = (
                        self._f_min_sum(R[i:i + s, j + 1], L[i:i + s, j + 1])
                        + L[i + s:i + 2 * s, j + 1]
                    )

            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    R[i:i + s, j] = self._f_min_sum(
                        R[i + s:i + 2 * s, j] + L[i + s:i + 2 * s, j],
                        R[i:i + s, j - 1],
                    )
                    R[i + s:i + 2 * s, j] = (
                        self._f_min_sum(R[i:i + s, j - 1], L[i:i + s, j])
                        + R[i + s:i + 2 * s, j]
                    )

            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(int)
            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(int)
        return u_hat, num_iters
