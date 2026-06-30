"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from decoder_sc import _bit_reversed, _prepare_llr, f_operation
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

    def _ms_f(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = _prepare_llr(llr_ch)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0

        frozen_l = np.zeros(N, dtype=bool)
        for phi in range(N):
            l = _bit_reversed(phi, n)
            frozen_l[l] = self.frozen_bits[l]
        R[frozen_l, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(self.max_iter):
            num_iters = it + 1

            for stage in range(n - 1, -1, -1):
                span = 1 << stage
                for block in range(0, N, span * 2):
                    for i in range(block, block + span):
                        j = i + span
                        L[i, stage] = self._ms_f(
                            R[i, stage + 1] + L[j, stage + 1], L[i, stage + 1]
                        )
                        L[j, stage] = (
                            self._ms_f(R[i, stage + 1], L[i, stage + 1])
                            + L[j, stage + 1]
                        )

            for stage in range(n):
                span = 1 << stage
                for block in range(0, N, span * 2):
                    for i in range(block, block + span):
                        j = i + span
                        R[i, stage + 1] = self._ms_f(
                            R[j, stage + 1] + L[j, stage + 1], R[i, stage]
                        )
                        R[j, stage + 1] = (
                            self._ms_f(R[i, stage], L[i, stage + 1]) + R[j, stage]
                        )

            for i in range(N):
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
            u_hat[frozen_l] = 0

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard):
                break

        for i in range(N):
            u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
        u_hat[frozen_l] = 0

        return u_hat, num_iters
