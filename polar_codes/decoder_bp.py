"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode_no_br


def _f_min_sum(x, y, alpha):
    sx = np.sign(x)
    sy = np.sign(y)
    sx = np.where(sx == 0, 1, sx)
    sy = np.where(sy == 0, 1, sy)
    return alpha * sx * sy * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.LARGE = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(step):
                    idx_u = np.arange(i, N, 2 * step)
                    idx_l = idx_u + step
                    L[idx_u, j - 1] = _f_min_sum(
                        R[idx_u, j - 1] + L[idx_l, j], L[idx_u, j], self.alpha
                    )
                    L[idx_l, j - 1] = _f_min_sum(
                        R[idx_u, j - 1], L[idx_u, j], self.alpha
                    ) + L[idx_l, j]

            for j in range(1, n + 1):
                step = 1 << (j - 1)
                for i in range(step):
                    idx_u = np.arange(i, N, 2 * step)
                    idx_l = idx_u + step
                    R[idx_u, j] = _f_min_sum(
                        R[idx_l, j - 1] + L[idx_l, j], R[idx_u, j - 1], self.alpha
                    )
                    R[idx_l, j] = _f_min_sum(
                        R[idx_u, j - 1], L[idx_u, j], self.alpha
                    ) + R[idx_l, j - 1]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode_no_br(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
