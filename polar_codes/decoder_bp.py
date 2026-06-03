"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
  """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        if 2**self.n != N:
            raise ValueError("N must be power of 2")
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)

    def _f_min_sum(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        for j in range(N):
            L[j, n] = llr_ch[self.br[j]]

        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = 1e6

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                span = 2 ** (n - j + 1)
                num_blocks = 2 ** (j - 1)
                for block in range(num_blocks):
                    base = block * span
                    half = span // 2
                    for i in range(half):
                        idx = base + i
                        idx2 = base + half + i
                        L[idx, j - 1] = self._f_min_sum(
                            R[idx, j] + L[idx2, j], L[idx, j]
                        )
                        L[idx2, j - 1] = self._f_min_sum(
                            R[idx, j], L[idx, j]
                        ) + L[idx2, j]

            for j in range(1, n + 1):
                span = 2 ** (n - j + 1)
                num_blocks = 2 ** (j - 1)
                for block in range(num_blocks):
                    base = block * span
                    half = span // 2
                    for i in range(half):
                        idx = base + i
                        idx2 = base + half + i
                        R[idx, j - 1] = self._f_min_sum(
                            R[idx2, j] + L[idx2, j], R[idx, j - 1]
                        )
                        R[idx2, j - 1] = self._f_min_sum(
                            R[idx, j - 1], L[idx, j]
                        ) + R[idx2, j]

            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break
        else:
            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1
            u_hat[self.frozen_bits] = 0

        return u_hat, num_iters
