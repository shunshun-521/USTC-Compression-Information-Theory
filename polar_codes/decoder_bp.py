"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import f_operation, _prepare_channel_llr
from encoder import polar_encode_matrix


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.LARGE = 1e6

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_nat = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = _prepare_channel_llr(llr_nat)
        N = self.N
        n = self.n
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        u_hat = np.zeros(N, dtype=int)
        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for r in range(s):
                        idx = i + r
                        if j + 1 > n:
                            continue
                        L[idx, j] = self._f_min_sum(
                            R[idx, j] + L[idx + s, j + 1],
                            L[idx, j + 1],
                        )
                        L[idx + s, j] = self._f_min_sum(R[idx, j], L[idx, j + 1]) + L[idx + s, j + 1]

            for j in range(1, n):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for r in range(s):
                        idx = i + r
                        R[idx, j] = self._f_min_sum(
                            R[idx + s, j] + L[idx + s, j + 1],
                            R[idx, j - 1],
                        )
                        R[idx + s, j] = self._f_min_sum(R[idx, j - 1], L[idx, j + 1]) + R[idx + s, j]

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode_matrix(u_hat)
            hard = (llr_nat < 0).astype(int)
            if np.array_equal(x_hat, hard):
                num_iters = it
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
