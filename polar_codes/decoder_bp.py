"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation

_LARGE = 1e6


class BPDecoder:
    """BP 译码器（因子图 min-sum）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.max_iter = max_iter
        self.alpha = alpha
        frozen_bits = np.asarray(frozen_bits)
        self.frozen_mask = frozen_bits != 0
        self.br = bit_reversal_permutation(N)
        self.inv_br = np.argsort(self.br)

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch[self.inv_br]
        R[:, 0] = 0.0
        for i in range(N):
            if self.frozen_mask[i]:
                R[i, 0] = _LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            for j in range(n - 1, -1, -1):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    Li, Ri = i, i + step
                    L[Li, j] = self._f_ms(R[Li, j] + L[Ri, j + 1], L[Li, j + 1])
                    L[Ri, j] = self._f_ms(R[Li, j], L[Li, j + 1]) + L[Ri, j + 1]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    Li, Ri = i, i + step
                    R[Li, j + 1] = self._f_ms(R[Ri, j] + L[Ri, j + 1], R[Li, j])
                    R[Ri, j + 1] = self._f_ms(R[Li, j], L[Li, j + 1]) + R[Ri, j]

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1
                if self.frozen_mask[i]:
                    u_hat[i] = 0

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard):
                num_iters = it
                break

        return u_hat, num_iters
