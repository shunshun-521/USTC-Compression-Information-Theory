"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, 0] = llr_ch
        R[:, 0] = 0.0
        frozen_idx = np.where(self.frozen_bits == 1)[0]
        R[frozen_idx, 0] = self.LARGE

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it

            for s in range(n):
                step = 1 << s
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        i0 = i + k
                        i1 = i + k + step
                        L[i0, s + 1] = self._f_ms(R[i0, s] + L[i1, s], L[i0, s])
                        L[i1, s + 1] = self._f_ms(R[i0, s], L[i0, s]) + L[i1, s]

            for s in range(n - 1, -1, -1):
                step = 1 << s
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        i0 = i + k
                        i1 = i + k + step
                        R[i0, s] = self._f_ms(R[i1, s + 1] + L[i1, s + 1], R[i0, s + 1])
                        R[i1, s] = self._f_ms(R[i0, s + 1], L[i0, s + 1]) + R[i1, s + 1]

            u_hat = self._hard_decode(L, R)
            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                break

        u_hat = self._hard_decode(L, R)
        return u_hat, num_iters

    def _hard_decode(self, L, R):
        u_hat = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
        return u_hat
