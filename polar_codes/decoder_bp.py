"""
极化码 BP（置信传播）译码器
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器：min-sum 近似 + 早停"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        if self.frozen.dtype != bool:
            self.frozen = self.frozen != 0
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _f_ms(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen, 0] = self.large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=np.int32)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx = i + k
                        s = idx + step
                        L[idx, j - 1] = self._f_ms(
                            R[idx, j] + L[s, j], L[idx, j + 1]
                        )
                        L[s, j - 1] = self._f_ms(R[idx, j], L[idx, j + 1]) + L[s, j + 1]

            for j in range(1, n + 1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx = i + k
                        s = idx + step
                        R[idx, j] = self._f_ms(
                            R[s, j] + L[s, j], R[idx, j - 1]
                        )
                        R[s, j] = self._f_ms(R[idx, j - 1], L[idx, j + 1]) + R[s, j]

            for i in range(N):
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
            u_hat[self.frozen] = 0

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(np.int32)
            if np.array_equal(x_hat, hard):
                num_iters = it
                break

        return u_hat, num_iters
