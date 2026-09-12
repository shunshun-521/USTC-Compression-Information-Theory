"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from decoder_sc import f_operation, clip_llr
from encoder import polar_encode


class BPDecoder:
    """
    BP 译码器。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        if 2 ** self.n != N:
            raise ValueError("N must be a power of 2")
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _f_ms(self, a, b):
        return clip_llr(
            self.alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))
        )

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 (u_hat, num_iters)。
        """
        llr_ch = clip_llr(np.asarray(llr_ch, dtype=np.float64))
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(i, i + s):
                        L[k, j - 1] = self._f_ms(
                            R[k, j] + L[k + s, j], L[k, j]
                        )
                        L[k + s, j - 1] = self._f_ms(
                            R[k, j], L[k, j]
                        ) + L[k + s, j]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(i, i + s):
                        R[k, j + 1] = self._f_ms(
                            R[k + s, j] + L[k + s, j + 1], R[k, j]
                        )
                        R[k + s, j + 1] = self._f_ms(
                            R[k, j], L[k, j + 1]
                        ) + R[k + s, j]

            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                return u_hat, it

        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(int)
        return u_hat, self.max_iter
