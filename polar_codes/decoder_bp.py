"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode
from channel import hard_decision_llr
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375, llr_max=1e10):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.llr_max = llr_max
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]

    def _f_min_sum(self, a, b):
        a = np.clip(a, -self.llr_max, self.llr_max)
        b = np.clip(b, -self.llr_max, self.llr_max)
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        large = self.llr_max

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n] = llr_ch
        R[0] = 0.0
        R[0, self.frozen_idx] = large

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for stage in range(n - 1, -1, -1):
                step = 1 << stage
                for i in range(0, N, 2 * step):
                    for j in range(i, i + step):
                        L[stage, j] = self._f_min_sum(
                            L[stage + 1, j],
                            L[stage + 1, j + step] + R[stage, j + step],
                        )
                        L[stage, j + step] = (
                            self._f_min_sum(R[stage, j], L[stage + 1, j])
                            + L[stage + 1, j + step]
                        )

            for stage in range(0, n):
                step = 1 << stage
                for i in range(0, N, 2 * step):
                    for j in range(i, i + step):
                        R[stage + 1, j] = self._f_min_sum(
                            R[stage, j],
                            L[stage + 1, j + step] + R[stage, j + step],
                        )
                        R[stage + 1, j + step] = (
                            self._f_min_sum(R[stage, j], L[stage + 1, j])
                            + R[stage, j + step]
                        )

            total = L[0] + R[0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[self.info_idx] = (total[self.info_idx] < 0).astype(int)

            if np.array_equal(polar_encode(u_hat), hard_decision_llr(llr_ch)):
                num_iters = it
                return u_hat, num_iters

        u_hat = np.zeros(N, dtype=int)
        total = L[0] + R[0]
        u_hat[self.info_idx] = (total[self.info_idx] < 0).astype(int)
        return u_hat, num_iters
