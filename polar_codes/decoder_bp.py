"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（极化因子图）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.max_iter = max_iter
        self.alpha = alpha
        fb = np.asarray(frozen_bits)
        self.frozen = fb.astype(bool) if fb.dtype == bool else (fb != 0)

    def _f_ms(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N
        large = 1e6
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen, 0] = large

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it
            for j in range(n - 1, -1, -1):
                step = 1 << j
                for i in range(0, N, step * 2):
                    for t in range(step):
                        idx = i + t
                        La = R[idx, j] + L[idx + step, j + 1]
                        Lb = L[idx, j + 1]
                        L[idx, j] = self._f_ms(La, Lb)
                        Rb = R[idx, j]
                        Lc = L[idx + step, j + 1]
                        L[idx + step, j] = self._f_ms(Rb, Lc) + L[idx + step, j + 1]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, step * 2):
                    for t in range(step):
                        idx = i + t
                        Ra = R[idx + step, j] + L[idx + step, j + 1]
                        Rb = R[idx, j]
                        R[idx, j + 1] = self._f_ms(Ra, Rb)
                        Rb2 = R[idx, j]
                        Lc = L[idx, j + 1]
                        R[idx + step, j + 1] = self._f_ms(Rb2, Lc) + R[idx + step, j]

            u_hat = np.zeros(N, dtype=int)
            total = L[:, 0] + R[:, 0]
            u_hat[total < 0] = 1
            u_hat[self.frozen] = 0

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                break

        u_hat = np.zeros(N, dtype=int)
        total = L[:, 0] + R[:, 0]
        u_hat[total < 0] = 1
        u_hat[self.frozen] = 0
        return u_hat, num_iters
