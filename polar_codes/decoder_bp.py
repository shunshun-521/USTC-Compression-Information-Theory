"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import f_operation, _prepare_channel_llr
from encoder import polar_encode


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = _prepare_channel_llr(llr_ch)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self._large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                span = 1 << (j - 1)
                for i in range(0, N, 2 * span):
                    for k in range(span):
                        left = i + k
                        right = i + k + span
                        L[left, j - 1] = self._f_min_sum(
                            R[left, j - 1] + L[right, j], L[left, j]
                        )
                        L[right, j - 1] = self._f_min_sum(
                            R[left, j - 1], L[left, j]
                        ) + L[right, j]

            for j in range(0, n):
                span = 1 << j
                for i in range(0, N, 2 * span):
                    for k in range(span):
                        left = i + k
                        right = i + k + span
                        R[left, j + 1] = self._f_min_sum(
                            R[right, j] + L[right, j + 1], R[left, j]
                        )
                        R[right, j + 1] = self._f_min_sum(
                            R[left, j], L[left, j + 1]
                        ) + R[right, j]

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            u_hat[i] = 0 if total >= 0 else 1
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
