"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def _f_min_sum(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0

        for it in range(self.max_iter):
            num_iters = it + 1

            for j in range(n, 0, -1):
                step = 2 ** (j - 1)
                for i in range(0, N, 2 * step):
                    L[i, j - 1] = self._f_min_sum(
                        R[i, j] + L[i + step, j],
                        L[i, j],
                    )
                    L[i + step, j - 1] = (
                        self._f_min_sum(R[i, j], L[i, j])
                        + L[i + step, j]
                    )

            for j in range(0, n):
                step = 2 ** j
                for i in range(0, N, 2 * step):
                    R[i, j + 1] = self._f_min_sum(
                        R[i + step, j] + L[i + step, j + 1],
                        R[i, j],
                    )
                    R[i + step, j + 1] = (
                        self._f_min_sum(R[i, j], L[i + step, j + 1])
                        + R[i + step, j]
                    )

            total_llr = L[:, 0] + R[:, 0]
            u_hat = self._hard_decision(total_llr)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        total_llr = L[:, 0] + R[:, 0]
        u_hat = self._hard_decision(total_llr)
        return u_hat, num_iters

    def _hard_decision(self, total_llr):
        u_hat = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total_llr[i] >= 0 else 1
        return u_hat
