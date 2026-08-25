"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import f_operation, prepare_decoder_llr
from encoder import bit_reversal_permutation, polar_encode


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = prepare_decoder_llr(np.asarray(llr_ch, dtype=np.float64))
        n = self.n
        N = self.N

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_bits] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for layer in range(n, 0, -1):
                step = 1 << (layer - 1)
                for block in range(0, N, 2 * step):
                    for j in range(block, block + step):
                        L[layer - 1][j] = self._f_min_sum(
                            R[layer][j] + L[layer][j + step], L[layer][j]
                        )
                        L[layer - 1][j + step] = self._f_min_sum(
                            R[layer][j], L[layer][j]
                        ) + L[layer][j + step]

            for layer in range(0, n):
                step = 1 << layer
                for block in range(0, N, 2 * step):
                    for j in range(block, block + step):
                        R[layer + 1][j] = self._f_min_sum(
                            R[layer + 1][j + step] + L[layer + 1][j + step],
                            R[layer][j],
                        )
                        R[layer + 1][j + step] = self._f_min_sum(
                            R[layer][j], L[layer + 1][j]
                        ) + R[layer + 1][j + step]

            for i in range(N):
                total = L[0][i] + R[0][i]
                u_hat[i] = 0 if self.frozen_bits[i] or total >= 0 else 1

            x_hat = polar_encode(u_hat)
            br = bit_reversal_permutation(N)
            hard = np.zeros(N, dtype=int)
            for i in range(N):
                hard[br[i]] = 0 if llr_ch[i] >= 0 else 1
            if np.array_equal(x_hat, hard):
                num_iters = it
                break

        for i in range(N):
            total = L[0][i] + R[0][i]
            u_hat[i] = 0 if self.frozen_bits[i] or total >= 0 else 1

        return u_hat, num_iters
