"""
极化码 BP（置信传播）译码器：min-sum + 早停
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """基于极化因子图的 BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        # L[i,j]: 右向左消息，j=n 为信道 LLR
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self._large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for block in range(0, N, 2 * s):
                    for i in range(s):
                        idx = block + i
                        L[idx, j] = self._f_ms(
                            R[idx, j + 1] + L[idx + s, j + 1], L[idx, j + 1]
                        )
                        L[idx + s, j] = self._f_ms(R[idx, j + 1], L[idx, j + 1]) + L[
                            idx + s, j + 1
                        ]

            for j in range(0, n):
                s = 1 << j
                for block in range(0, N, 2 * s):
                    for i in range(s):
                        idx = block + i
                        R[idx, j + 1] = self._f_ms(
                            R[idx + s, j] + L[idx + s, j + 1], R[idx, j]
                        )
                        R[idx + s, j + 1] = self._f_ms(R[idx, j], L[idx, j + 1]) + R[
                            idx + s, j
                        ]

            total = L[:, 0] + R[:, 0]
            u_hat = np.where(total >= 0, 0, 1).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.where(total >= 0, 0, 1).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
