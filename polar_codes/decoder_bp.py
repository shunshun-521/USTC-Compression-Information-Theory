"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6
        self._layers = []
        for j in range(1, self.n + 1):
            s = 1 << (j - 1)
            idx = np.arange(0, N, 2 * s)[:, None] + np.arange(s)
            self._layers.append((j, idx.ravel(), (idx + s).ravel()))

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self._large

        num_iters = self.max_iter
        hard_ch = (llr_ch < 0).astype(int)

        for it in range(1, self.max_iter + 1):
            for j, lo, hi in self._layers[::-1]:
                L[lo, j - 1] = self._f_min_sum(
                    R[lo, j - 1] + L[hi, j], L[lo, j]
                )
                L[hi, j - 1] = self._f_min_sum(R[lo, j - 1], L[lo, j]) + L[hi, j]

            for j, lo, hi in self._layers:
                R[lo, j] = self._f_min_sum(
                    R[hi, j - 1] + L[hi, j], R[lo, j - 1]
                )
                R[hi, j] = self._f_min_sum(R[lo, j - 1], L[lo, j]) + R[hi, j - 1]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            if np.array_equal(polar_encode(u_hat), hard_ch):
                num_iters = it
                break
            num_iters = it

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
