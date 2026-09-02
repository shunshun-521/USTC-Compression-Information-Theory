"""
极化码 BP（置信传播）译码器
基于因子图（列 0=信道，列 n=信源），min-sum 近似，含早停
"""
import math

import numpy as np

from encoder import polar_encode


def _sign_pm(x):
    return np.where(x >= 0, 1.0, -1.0)


def _minsum_f(a, b, alpha):
    return alpha * _sign_pm(a) * _sign_pm(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha

    def _init_messages(self, llr_ch):
        L = np.zeros((self.n + 1, self.N), dtype=np.float64)
        R = np.zeros((self.n + 1, self.N), dtype=np.float64)
        L[0, :] = llr_ch
        info_idx = np.where(self.frozen_bits == 0)[0]
        R[self.n, info_idx] = 0.0
        R[self.n, self.frozen_bits == 1] = self.LARGE
        return L, R

    def _update_left(self, L, R):
        for i in range(self.n):
            half = 1 << i
            block_size = half << 1
            for block in range(0, self.N, block_size):
                for k in range(half):
                    j0 = block + k
                    j1 = block + k + half
                    j2 = block + 2 * k
                    j3 = j2 + 1
                    L[i + 1, j2] = _minsum_f(
                        R[i + 1, j3] + L[i, j1], L[i, j0], self.alpha
                    )
                    L[i + 1, j3] = _minsum_f(
                        R[i + 1, j2], L[i, j0], self.alpha
                    ) + L[i, j1]

    def _update_right(self, L, R):
        for i in range(self.n - 1, -1, -1):
            half = 1 << i
            block_size = half << 1
            for block in range(0, self.N, block_size):
                for k in range(half):
                    j0 = block + k
                    j1 = block + k + half
                    j2 = block + 2 * k
                    j3 = j2 + 1
                    R[i, j0] = _minsum_f(
                        R[i + 1, j2], R[i + 1, j3] + L[i, j1], self.alpha
                    )
                    R[i, j1] = R[i + 1, j3] + _minsum_f(
                        R[i + 1, j2], L[i, j0], self.alpha
                    )

    def _hard_decode(self, L, R):
        total = L[self.n, :] + R[self.n, :]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits == 1] = 0
        return u_hat

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        L, R = self._init_messages(llr_ch)

        num_iters = self.max_iter
        u_hat = None
        for it in range(1, self.max_iter + 1):
            self._update_left(L, R)
            self._update_right(L, R)
            u_hat = self._hard_decode(L, R)
            x_hat = polar_encode(u_hat)
            channel_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, channel_hard):
                num_iters = it
                break

        if u_hat is None:
            u_hat = self._hard_decode(L, R)
        return u_hat, num_iters
