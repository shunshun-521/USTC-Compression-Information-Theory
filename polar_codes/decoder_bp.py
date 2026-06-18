"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _f_minsum(x, y, alpha):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    sx, sy = np.sign(x), np.sign(y)
    sx = np.where(sx == 0, 1, sx)
    sy = np.where(sy == 0, 1, sy)
    return alpha * sx * sy * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def decode(self, llr_ch):
        """主译码函数"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        L = np.zeros((self.N, self.n + 1), dtype=np.float64)
        R = np.zeros((self.N, self.n + 1), dtype=np.float64)
        L[:, self.n] = llr_ch[self.br]
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(self.N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            self._update_left(L, R)
            self._update_right(L, R)

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters

    def _update_left(self, L, R):
        alpha = self.alpha
        for j in range(self.n - 1, -1, -1):
            step = 1 << j
            for i in range(0, self.N, 2 * step):
                for k in range(step):
                    a = i + k
                    b = a + step
                    L[a, j] = _f_minsum(
                        R[a, j] + L[b, j + 1], L[a, j + 1], alpha
                    )
                    L[b, j] = _f_minsum(R[a, j], L[a, j + 1], alpha) + L[b, j + 1]

    def _update_right(self, L, R):
        alpha = self.alpha
        for j in range(0, self.n):
            step = 1 << j
            for i in range(0, self.N, 2 * step):
                for k in range(step):
                    a = i + k
                    b = a + step
                    R[a, j + 1] = _f_minsum(
                        R[b, j] + L[b, j + 1], R[a, j], alpha
                    )
                    R[b, j + 1] = _f_minsum(R[a, j], L[a, j + 1], alpha) + R[b, j]
