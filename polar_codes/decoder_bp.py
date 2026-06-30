"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def _g(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        channel_llr = L[:, n].copy()
        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for c in range(n - 1, -1, -1):
                block_size = 1 << (n - c)
                offset = block_size // 2
                for bs in range(0, N, block_size):
                    for k in range(offset):
                        j = bs + k
                        jp = j + offset
                        L[j, c] = self._g(L[j, c + 1], L[jp, c + 1] + R[jp, c])
                        L[jp, c] = self._g(L[j, c + 1], R[j, c]) + L[jp, c + 1]

            for c in range(0, n):
                block_size = 1 << (n - c)
                offset = block_size // 2
                for bs in range(0, N, block_size):
                    for k in range(offset):
                        j = bs + k
                        jp = j + offset
                        R[j, c + 1] = self._g(R[j, c], L[jp, c + 1] + R[jp, c])
                        R[jp, c + 1] = self._g(L[j, c + 1], R[j, c]) + R[jp, c]

            L[:, n] = channel_llr

            u_hat = self._hard_decision(L, R)
            if self._early_stop(u_hat, llr_ch):
                num_iters = it
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters

    def _hard_decision(self, L, R):
        u_hat = (L[:, 0] < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat

    def _early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, hard_ch)
