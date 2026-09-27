"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode


class BPDecoder:
    """BP 译码器（极化码因子图）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6

    def _f_ms(self, a, b):
        return self.alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        rev = np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)
        L[:, n] = llr_ch[rev]
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self._large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        Rv = R[idx, j - 1]
                        Lup = L[idx + s, j]
                        Ldn = L[idx, j]
                        Rdn = R[idx + s, j - 1]
                        Lup2 = L[idx, j]
                        Rup = R[idx, j - 1]

                        L[idx, j - 1] = self._f_ms(Rv + Lup, Ldn)
                        L[idx + s, j - 1] = self._f_ms(Rup, Lup2) + L[idx + s, j]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        Rdn = R[idx + s, j]
                        Ldn = L[idx + s, j + 1]
                        Rup = R[idx, j]
                        Lup = L[idx, j + 1]
                        Rv = R[idx + s, j]

                        R[idx, j + 1] = self._f_ms(Rdn + Ldn, Rup)
                        R[idx + s, j + 1] = self._f_ms(Rup, Lup) + Rv

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat.astype(int), num_iters
