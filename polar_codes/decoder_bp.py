"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import _prepare_channel_llr, f_operation
from encoder import polar_encode
from channel import hard_decision_llr


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=np.int8)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.large = 1e6

    def _minsum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = _prepare_channel_llr(llr_ch)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for j in range(n - 1, -1, -1):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx = i + k
                        idx_s = idx + step
                        L[idx, j] = self._minsum(
                            R[idx, j] + L[idx_s, j + 1], L[idx, j + 1]
                        )
                        L[idx_s, j] = self._minsum(
                            R[idx, j], L[idx, j + 1]
                        ) + L[idx_s, j + 1]

            for j in range(n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx = i + k
                        idx_s = idx + step
                        R[idx, j + 1] = self._minsum(
                            R[idx_s, j] + L[idx_s, j + 1], R[idx, j]
                        )
                        R[idx_s, j + 1] = self._minsum(
                            R[idx, j], L[idx, j + 1]
                        ) + R[idx_s, j]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(np.int8)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(np.int8)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
