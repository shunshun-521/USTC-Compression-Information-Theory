"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器（min-sum）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_ms(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_ch
        for i in range(N):
            R[0, i] = 0.0 if not self.frozen_bits[i] else self.LARGE

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for stage in range(n - 1, -1, -1):
                step = 1 << stage
                for block in range(0, N, 2 * step):
                    for t in range(step):
                        i = block + t
                        j = i + step
                        L[stage, i] = self._f_ms(
                            R[stage + 1, i] + L[stage + 1, j], L[stage + 1, i]
                        )
                        L[stage, j] = self._f_ms(R[stage + 1, i], L[stage + 1, i]) + L[
                            stage + 1, j
                        ]

            for stage in range(n):
                step = 1 << stage
                for block in range(0, N, 2 * step):
                    for t in range(step):
                        i = block + t
                        j = i + step
                        R[stage + 1, i] = self._f_ms(
                            R[stage + 1, j] + L[stage + 1, j], R[stage, i]
                        )
                        R[stage + 1, j] = self._f_ms(R[stage, i], L[stage + 1, i]) + R[
                            stage + 1, j
                        ]

            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                total = L[0, i] + R[0, i]
                u_hat[i] = 0 if total >= 0 else 1
                if self.frozen_bits[i]:
                    u_hat[i] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
        else:
            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                total = L[0, i] + R[0, i]
                u_hat[i] = 0 if total >= 0 else 1
                if self.frozen_bits[i]:
                    u_hat[i] = 0

        return u_hat, num_iters
