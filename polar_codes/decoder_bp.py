"""
极化码 BP（置信传播）译码器
"""
import numpy as np
from decoder_sc import f_operation, upper_llr, lower_llr
from encoder import polar_encode


class BPDecoder:
    """BP 译码器（min-sum + 早停）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e8

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, 0] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it
            for s in range(n):
                block = 1 << (s + 1)
                half = block // 2
                for j in range(0, N, block):
                    for k in range(half):
                        i = j + k
                        L[i, s + 1] = self._f_ms(
                            R[i, s] + L[i + half, s], L[i + half, s]
                        )
                        L[i + half, s + 1] = self._f_ms(R[i, s], L[i, s]) + L[i + half, s]

            for s in range(n - 1, -1, -1):
                block = 1 << (s + 1)
                half = block // 2
                for j in range(0, N, block):
                    for k in range(half):
                        i = j + k
                        R[i, s + 1] = self._f_ms(
                            R[i + half, s] + L[i + half, s + 1], R[i, s]
                        )
                        R[i + half, s + 1] = self._f_ms(R[i, s], L[i, s + 1]) + R[i + half, s]

            total = L[:, n] + R[:, n]
            u_hat = np.zeros(N, dtype=np.int8)
            u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(np.int8)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard_ch):
                break

        total = L[:, n] + R[:, n]
        u_hat = np.zeros(N, dtype=np.int8)
        u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(np.int8)
        return u_hat, num_iters
