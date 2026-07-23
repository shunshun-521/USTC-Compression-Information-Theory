"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode
from decoder_sc import _prepare_channel_llrs


class BPDecoder:
    """BP 译码器（参考 MDPI Symmetry 2022 因子图索引）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e10

    def _g(self, a, b):
        return self.alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, num_iters。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = _prepare_channel_llrs(llr_ch)
        R[0, self.frozen_bits] = self.LARGE

        num_iters = 0
        for it in range(self.max_iter):
            num_iters = it + 1

            for i in range(n - 1, -1, -1):
                offset = N >> (i + 1)
                block = offset << 1
                for base in range(0, N, block):
                    for j in range(offset):
                        idx1 = base + j
                        idx2 = idx1 + offset
                        L[i, idx1] = self._g(
                            L[i + 1, idx1],
                            L[i + 1, idx2] + R[i, idx2],
                        )
                        L[i, idx2] = self._g(L[i + 1, idx1], R[i, idx1]) + L[i + 1, idx2]

            for i in range(0, n):
                offset = N >> (i + 1)
                block = offset << 1
                for base in range(0, N, block):
                    for j in range(offset):
                        idx1 = base + j
                        idx2 = idx1 + offset
                        R[i + 1, idx1] = self._g(
                            R[i, idx1],
                            L[i + 1, idx2] + R[i, idx2],
                        )
                        R[i + 1, idx2] = self._g(L[i + 1, idx1], R[i, idx1]) + R[i, idx2]

            total = L[0, :] + R[0, :]
            u_hat = np.zeros(N, dtype=int)
            u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(int)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        total = L[0, :] + R[0, :]
        u_hat = np.zeros(N, dtype=int)
        u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(int)
        return u_hat, num_iters
