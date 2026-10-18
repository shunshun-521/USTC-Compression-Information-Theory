"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（min-sum，冻结位强先验）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self._large = 1e6

    def _f(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_ch[self.br]
        R[0, :] = 0.0
        R[0, self.frozen_bits] = self._large

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            # L：从右到左
            for lam in range(n - 1, -1, -1):
                step = 1 << lam
                for i in range(0, N, 2 * step):
                    for j in range(step):
                        a = i + j
                        b = a + step
                        L[lam, a] = self._f(R[lam + 1, a] + L[lam + 1, b], L[lam + 1, a])
                        L[lam, b] = self._f(R[lam + 1, a], L[lam + 1, a]) + L[lam + 1, b]

            # R：从左到右
            for lam in range(1, n + 1):
                step = 1 << (lam - 1)
                for i in range(0, N, 2 * step):
                    for j in range(step):
                        a = i + j
                        b = a + step
                        R[lam, a] = self._f(R[lam - 1, b] + L[lam, b], R[lam - 1, a])
                        R[lam, b] = self._f(R[lam - 1, a], L[lam, a]) + R[lam - 1, b]

            total = L[0, :] + R[0, :]
            u_hat = np.zeros(N, dtype=int)
            u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(int)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[0, :] + R[0, :]
        u_hat = np.zeros(N, dtype=int)
        u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(int)
        return u_hat, num_iters
