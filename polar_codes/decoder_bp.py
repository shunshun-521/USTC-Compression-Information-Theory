"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import _frozen_mask_to_mcba1n_set


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.br = bit_reversal_permutation(N)
        self.frozen_set = _frozen_mask_to_mcba1n_set(frozen_bits, self.br)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _f_min_sum(self, x, y):
        return self.alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))

    def _hard_decode(self, L, R):
        u_mcba = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            if i in self.frozen_set:
                u_mcba[i] = 0
            else:
                u_mcba[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
        return u_mcba[self.br]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        for idx in self.frozen_set:
            R[idx, 0] = self.large

        u_hat = np.zeros(N, dtype=int)
        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    L[i, j - 1] = self._f_min_sum(
                        R[i, j - 1] + L[i + step, j], L[i, j]
                    )
                    L[i + step, j - 1] = (
                        self._f_min_sum(R[i, j - 1], L[i, j]) + L[i + step, j]
                    )

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    R[i, j + 1] = self._f_min_sum(
                        R[i + step, j] + L[i + step, j + 1], R[i, j]
                    )
                    R[i + step, j + 1] = (
                        self._f_min_sum(R[i, j], L[i, j + 1]) + R[i + step, j]
                    )

            u_hat = self._hard_decode(L, R)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        return u_hat, num_iters
