"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import bit_reversal_permutation, polar_encode
from decoder_sc import f_operation


LARGE = 1e6


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.br = bit_reversal_permutation(N)

    def _f(self, a, b):
        return self.alpha * f_operation(a, b)

    def _hard_decision(self, L, R):
        total = L + R
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat

    def _update_llr(self, L, R):
        n = self.n
        N = self.N
        for j in range(n - 1, -1, -1):
            s = 1 << j
            for block in range(0, N, 2 * s):
                for i in range(block, block + s):
                    L[i, j] = self._f(
                        L[i, j + 1],
                        R[i + s, j] + L[i + s, j + 1],
                    )
                    L[i + s, j] = self._f(R[i, j], L[i, j + 1]) + L[i + s, j + 1]

    def _update_reflected(self, L, R):
        n = self.n
        N = self.N
        for j in range(n):
            s = 1 << j
            for block in range(0, N, 2 * s):
                for i in range(block, block + s):
                    R[i, j + 1] = self._f(
                        R[i, j],
                        L[i + s, j + 1] + R[i + s, j],
                    )
                    R[i + s, j + 1] = self._f(R[i, j], L[i, j + 1]) + R[i + s, j]

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, num_iters。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        llr_internal = llr_ch[self.br]

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_internal
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = LARGE

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            self._update_llr(L, R)
            self._update_reflected(L, R)

            u_hat = self._hard_decision(L[:, 0], R[:, 0])
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        u_hat = self._hard_decision(L[:, 0], R[:, 0])
        return u_hat, num_iters
