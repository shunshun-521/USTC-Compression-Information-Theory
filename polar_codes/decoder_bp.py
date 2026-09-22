"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from decoder_sc import _as_frozen_mask
from encoder import polar_encode


LARGE = 1e6


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（分层因子图）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = _as_frozen_mask(frozen_bits)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, num_iters。"""
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, s << 1):
                    L[i, j - 1] = _f_min_sum(
                        R[i, j] + L[i + s, j], L[i, j], self.alpha
                    )
                    L[i + s, j - 1] = _f_min_sum(
                        R[i, j], L[i, j], self.alpha
                    ) + L[i + s, j]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, s << 1):
                    R[i, j + 1] = _f_min_sum(
                        R[i + s, j] + L[i + s, j + 1], R[i, j], self.alpha
                    )
                    R[i + s, j + 1] = _f_min_sum(
                        R[i, j], L[i, j + 1], self.alpha
                    ) + R[i + s, j]

            num_iters = it

            posterior = L[:, 0] + R[:, 0]
            u_hat = (posterior < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        posterior = L[:, 0] + R[:, 0]
        u_hat = (posterior < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
