"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（因子图 n+1 列，min-sum + 早停）。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_nat = llr_ch[self.br]
        n = self.n
        N = self.N

        # L[i][j]: 行 i，列 j（0=信源端，n=信道端）
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_nat
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        frozen = self.frozen_bits
        num_iters = self.max_iter
        x_hard = (llr_ch < 0).astype(np.int32)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for base in range(0, N, 2 * s):
                    u = np.arange(base, base + s)
                    v = u + s
                    L[u, j - 1] = self._f_min_sum(R[u, j] + L[v, j], L[u, j])
                    L[v, j - 1] = self._f_min_sum(R[u, j], L[u, j]) + L[v, j]

            for j in range(0, n):
                s = 1 << j
                for base in range(0, N, 2 * s):
                    u = np.arange(base, base + s)
                    v = u + s
                    R[u, j + 1] = self._f_min_sum(R[v, j] + L[v, j + 1], R[u, j])
                    R[v, j + 1] = self._f_min_sum(R[u, j], L[u, j + 1]) + R[v, j]

            total = L[:, 0] + R[:, 0]
            u_hat = np.where(frozen, 0, (total < 0).astype(np.int32))
            if np.array_equal(polar_encode(u_hat), x_hard):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.where(frozen, 0, (total < 0).astype(np.int32))
        return u_hat, num_iters
