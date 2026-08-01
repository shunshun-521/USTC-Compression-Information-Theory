"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.rev = bit_reversal_permutation(N)

    def _f_minsum(self, a, b):
        return self.alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        llr_perm = llr_ch[self.rev]

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_perm
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = 1e6

        num_iters = self.max_iter

        for it in range(self.max_iter):
            for j in range(n, 0, -1):
                step = 2 ** (j - 1)
                for i in range(0, N, 2 * step):
                    L[i, j - 1] = self._f_minsum(
                        R[i, j] + L[i + step, j], L[i, j]
                    )
                    L[i + step, j - 1] = self._f_minsum(R[i, j], L[i, j]) + L[i + step, j]

            for j in range(1, n + 1):
                step = 2 ** (j - 1)
                for i in range(0, N, 2 * step):
                    R[i, j] = self._f_minsum(
                        R[i + step, j] + L[i + step, j], R[i, j - 1]
                    )
                    R[i + step, j] = self._f_minsum(R[i, j - 1], L[i, j]) + R[i + step, j]

            total = L[:, 0] + R[:, 0]
            u_hat = np.where(total >= 0, 0, 1).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_perm < 0).astype(int)
            x_perm = x_hat[self.rev]
            if np.array_equal(x_perm, hard_ch):
                num_iters = it + 1
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.where(total >= 0, 0, 1).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
