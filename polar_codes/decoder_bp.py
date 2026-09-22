"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import _to_frozen_mask, f_operation, g_operation


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = _to_frozen_mask(frozen_bits)
        self.max_iter = max_iter
        self.alpha = alpha
        self.rev = bit_reversal_permutation(N)

    def _f(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        n, N = self.n, self.N
        llr = llr_ch[self.rev].astype(np.float64)

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n] = llr
        R[0] = 0.0
        R[0, self.frozen_bits] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for stage in range(n - 1, -1, -1):
                stride = 2 ** (n - 1 - stage)
                for i in range(0, N, 2 * stride):
                    for j in range(stride):
                        idx1 = i + j
                        idx2 = i + j + stride
                        L[stage, idx1] = self._f(L[stage + 1, idx1], L[stage + 1, idx2] + R[stage, idx2])
                        L[stage, idx2] = self._f(L[stage + 1, idx1], R[stage, idx1]) + L[stage + 1, idx2]

            for stage in range(1, n + 1):
                stride = 2 ** (n - stage)
                for i in range(0, N, 2 * stride):
                    for j in range(stride):
                        idx1 = i + j
                        idx2 = i + j + stride
                        R[stage, idx2] = self._f(R[stage - 1, idx1], L[stage, idx1]) + R[stage - 1, idx2]
                        R[stage, idx1] = self._f(R[stage - 1, idx2] + L[stage, idx2], R[stage - 1, idx1])

            for i in range(N):
                total = L[0, i] + R[0, i]
                u_hat[i] = 0 if total >= 0 else 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_x = (llr < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break

        for i in range(N):
            total = L[0, i] + R[0, i]
            u_hat[i] = 0 if total >= 0 else 1
        u_hat[self.frozen_bits] = 0

        return u_hat, num_iters
