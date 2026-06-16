"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode, bit_reversal_permutation


class BPDecoder:
    """BP 译码器（min-sum，含早停）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.br = bit_reversal_permutation(N)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _minsum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits[self.br], 0] = self.large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 2 ** (j - 1)
                for i in range(0, N, 2 * s):
                    L[i, j - 1] = self._minsum(R[i, j] + L[i + s, j], L[i, j])
                    L[i + s, j - 1] = self._minsum(R[i, j], L[i, j]) + L[i + s, j]

            for j in range(0, n):
                s = 2 ** j
                for i in range(0, N, 2 * s):
                    R[i, j + 1] = self._minsum(R[i + s, j] + L[i + s, j + 1], R[i, j])
                    R[i + s, j + 1] = self._minsum(R[i, j], L[i, j + 1]) + R[i + s, j]

            total = L[:, 0] + R[:, 0]
            u_br = (total < 0).astype(int)
            u_br[self.frozen_bits[self.br]] = 0
            u_hat = np.empty(N, dtype=int)
            u_hat[self.br] = u_br

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break
        else:
            total = L[:, 0] + R[:, 0]
            u_br = (total < 0).astype(int)
            u_br[self.frozen_bits[self.br]] = 0
            u_hat = np.empty(N, dtype=int)
            u_hat[self.br] = u_br

        return u_hat, num_iters
