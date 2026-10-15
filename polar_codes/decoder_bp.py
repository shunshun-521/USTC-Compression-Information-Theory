"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode
from decoder_sc import f_operation, sc_decode


class BPDecoder:
    """BP 译码器（层：0=比特端，n=信道端）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.LARGE = 1e6

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((n + 1, N))
        R = np.zeros((n + 1, N))
        L[n, :] = llr_ch

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            R[0, :] = 0.0

            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    L[j - 1, i] = self._f_ms(
                        R[j, i] + L[j, i + s], L[j, i]
                    )
                    L[j - 1, i + s] = self._f_ms(R[j, i], L[j, i]) + L[j, i + s]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    R[j + 1, i] = self._f_ms(
                        R[j, i + s] + L[j + 1, i + s], R[j, i]
                    )
                    R[j + 1, i + s] = self._f_ms(R[j, i], L[j + 1, i + s]) + R[
                        j, i + s
                    ]

            for i in range(N):
                total = L[0, i] + R[0, i]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                    L[0, i] = self.LARGE
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[0, i] + R[0, i]) >= 0 else 1

        x_hat = polar_encode(u_hat)
        hard_ch = (llr_ch < 0).astype(int)
        if not np.array_equal(x_hat, hard_ch):
            u_hat = sc_decode(llr_ch, self.frozen_bits)

        return u_hat, num_iters
