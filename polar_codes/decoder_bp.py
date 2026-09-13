"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from decoder_sc import f_operation, _as_frozen_mask, _prepare_llr
from encoder import polar_encode


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = _as_frozen_mask(frozen_bits)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = _prepare_llr(llr_ch)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self._large

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                jp = min(j + 1, n)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx1 = i + k
                        idx2 = i + k + s
                        L[idx1, j - 1] = self._f_min_sum(
                            R[idx1, j] + L[idx2, jp], L[idx1, jp]
                        )
                        L[idx2, j - 1] = self._f_min_sum(
                            R[idx1, j], L[idx1, jp]
                        ) + L[idx2, jp]

            for j in range(0, n):
                s = 1 << j
                jp = min(j + 1, n)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx1 = i + k
                        idx2 = i + k + s
                        R[idx1, j + 1] = self._f_min_sum(
                            R[idx2, jp] + L[idx2, jp], R[idx1, j]
                        )
                        R[idx2, j + 1] = self._f_min_sum(
                            R[idx1, j], L[idx1, jp]
                        ) + R[idx2, j]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
