"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from decoder_sc import f_operation
from encoder import bit_reversal_permutation, polar_encode
from decoder_sc import _channel_llr_for_tree

_LARGE = 1e6


class BPDecoder:
    """BP 译码器（flooding，min-sum 带缩放因子 alpha）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.rev = bit_reversal_permutation(N)
        self.inv = np.argsort(self.rev)

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_perm = _channel_llr_for_tree(llr_ch)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_perm
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = _LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                stride = 1 << (j - 1)
                for block in range(0, N, 2 * stride):
                    for t in range(stride):
                        i = block + t
                        s = i + stride
                        L[i, j - 1] = self._f_ms(
                            R[i, j] + L[s, j], L[i, j]
                        )
                        L[s, j - 1] = self._f_ms(R[i, j], L[i, j]) + L[s, j]

            for j in range(0, n):
                stride = 1 << j
                for block in range(0, N, 2 * stride):
                    for t in range(stride):
                        i = block + t
                        s = i + stride
                        R[i, j + 1] = self._f_ms(
                            R[s, j] + L[s, j + 1], R[i, j]
                        )
                        R[s, j + 1] = self._f_ms(R[i, j], L[i, j + 1]) + R[s, j]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(np.int8)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(np.int8)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
