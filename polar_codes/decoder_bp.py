"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import f_operation
from encoder import bit_reversal_permutation, polar_encode

_LARGE = 1e6


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch[self.br]
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = _LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                stride = 1 << (j - 1)
                for block in range(0, N, 2 * stride):
                    for i in range(block, block + stride):
                        s = i + stride
                        L[i, j - 1] = self._f_min_sum(
                            R[i, j - 1] + L[s, j], L[i, j]
                        )
                        L[s, j - 1] = self._f_min_sum(
                            R[i, j - 1], L[i, j]
                        ) + L[s, j]

            for j in range(0, n):
                stride = 1 << j
                for block in range(0, N, 2 * stride):
                    for i in range(block, block + stride):
                        s = i + stride
                        R[i, j + 1] = self._f_min_sum(
                            R[s, j + 1] + L[s, j + 1], R[i, j]
                        )
                        R[s, j + 1] = self._f_min_sum(
                            R[i, j], L[s, j + 1]
                        ) + R[s, j + 1]

            num_iters = it
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    total = L[i, 0] + R[i, 0]
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
