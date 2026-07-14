"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation, _frozen_to_set


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = _frozen_to_set(frozen_bits)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self.large = 1e6

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        n = self.n
        N = self.N
        llr = np.asarray(llr_ch, dtype=np.float64)[self.br]

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr
        R[:, 0] = 0.0
        for idx in self.frozen:
            R[idx, 0] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int32)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, step * 2):
                    for k in range(step):
                        idx = i + k
                        s = step
                        L[idx, j - 1] = self._f_min_sum(
                            R[idx, j] + L[idx + s, j], L[idx, j]
                        )
                        L[idx + s, j - 1] = self._f_min_sum(
                            R[idx, j], L[idx, j]
                        ) + L[idx + s, j]

            for j in range(1, n + 1):
                step = 1 << (j - 1)
                for i in range(0, N, step * 2):
                    for k in range(step):
                        idx = i + k
                        s = step
                        R[idx, j] = self._f_min_sum(
                            R[idx + s, j] + L[idx + s, j], R[idx, j - 1]
                        )
                        R[idx + s, j] = (
                            self._f_min_sum(R[idx, j - 1], L[idx, j])
                            + R[idx + s, j - 1]
                        )

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1
            for idx in self.frozen:
                u_hat[idx] = 0

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(np.int32)
            if np.array_equal(x_hat, hard):
                break

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            u_hat[i] = 0 if total >= 0 else 1
        for idx in self.frozen:
            u_hat[idx] = 0

        return u_hat, num_iters
