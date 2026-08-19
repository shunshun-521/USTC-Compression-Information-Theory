"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from channel import reorder_llr_for_decode
from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        frozen_arr = np.asarray(frozen_bits)
        if frozen_arr.dtype == bool:
            self.frozen_bits = frozen_arr
        else:
            self.frozen_bits = frozen_arr.astype(int) == 1
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = reorder_llr_for_decode(llr_ch)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    L[i : i + step, j - 1] = self._f_min_sum(
                        R[i : i + step, j] + L[i + step : i + 2 * step, j],
                        L[i : i + step, j + 1],
                    )
                    L[i + step : i + 2 * step, j - 1] = self._f_min_sum(
                        R[i : i + step, j],
                        L[i : i + step, j + 1],
                    ) + L[i + step : i + 2 * step, j + 1]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    R[i : i + step, j + 1] = self._f_min_sum(
                        R[i + step : i + 2 * step, j] + L[i + step : i + 2 * step, j + 1],
                        R[i : i + step, j],
                    )
                    R[i + step : i + 2 * step, j + 1] = self._f_min_sum(
                        R[i : i + step, j],
                        L[i : i + step, j + 1],
                    ) + R[i + step : i + 2 * step, j]

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    total = L[i, 0] + R[i, 0]
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                num_iters = it
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
