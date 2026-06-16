"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import f_operation, _preprocess_llr
from encoder import bit_reversal_permutation, polar_encode


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.large = 1e6
        self._br_inv = np.argsort(bit_reversal_permutation(N))

    def _ms_f(self, a, b):
        return self.alpha * f_operation(a, b)

    def _hard_bits_channel(self, llr_ch):
        return (np.asarray(llr_ch, dtype=np.float64) < 0).astype(int)

    def decode(self, llr_ch):
        llr_raw = np.asarray(llr_ch, dtype=np.float64)
        llr_proc = _preprocess_llr(llr_raw)
        n, N = self.n, self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_proc
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)
        x_hard = self._hard_bits_channel(llr_raw)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        i0 = i + k
                        i1 = i + k + s
                        L[i0, j - 1] = self._ms_f(
                            R[i0, j] + L[i1, j], L[i0, j]
                        )
                        L[i1, j - 1] = self._ms_f(R[i0, j], L[i0, j]) + L[i1, j]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        i0 = i + k
                        i1 = i + k + s
                        R[i0, j + 1] = self._ms_f(
                            R[i1, j] + L[i1, j + 1], R[i0, j]
                        )
                        R[i1, j + 1] = self._ms_f(R[i0, j], L[i0, j + 1]) + R[i1, j]

            total = L[:, 0] + R[:, 0]
            for i in range(N):
                u_hat[i] = 0 if self.frozen_bits[i] else (0 if total[i] >= 0 else 1)

            if np.array_equal(polar_encode(u_hat), x_hard):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        for i in range(N):
            u_hat[i] = 0 if self.frozen_bits[i] else (0 if total[i] >= 0 else 1)

        return u_hat, num_iters
