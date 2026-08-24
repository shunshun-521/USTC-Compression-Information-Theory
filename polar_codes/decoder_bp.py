"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        m = self.n
        N = self.N

        L = np.zeros((N, m + 1), dtype=np.float64)
        R = np.zeros((N, m + 1), dtype=np.float64)
        L[:, m] = llr_ch
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        for it in range(self.max_iter):
            num_iters = it + 1

            for j in range(m - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    L[i : i + s, j] = self._f_ms(
                        L[i : i + s, j + 1],
                        L[i + s : i + 2 * s, j + 1] + R[i + s : i + 2 * s, j],
                    )
                    L[i + s : i + 2 * s, j] = (
                        self._f_ms(R[i : i + s, j], L[i : i + s, j + 1])
                        + L[i + s : i + 2 * s, j + 1]
                    )

            for j in range(1, m + 1):
                s = 1 << (m - j)
                for i in range(0, N, 2 * s):
                    R[i : i + s, j] = self._f_ms(
                        R[i + s : i + 2 * s, j - 1] + L[i + s : i + 2 * s, j],
                        R[i : i + s, j - 1],
                    )
                    R[i + s : i + 2 * s, j] = (
                        self._f_ms(R[i : i + s, j - 1], L[i : i + s, j])
                        + R[i + s : i + 2 * s, j - 1]
                    )

            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(int)
        u_hat[self.frozen_bits] = 0

        return u_hat, num_iters
