"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math
from encoder import polar_encode
from decoder_sc import f_operation_min_sum


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f(self, a, b):
        return self.alpha * f_operation_min_sum(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_bits] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        Li = i + k
                        Li_s = i + k + s
                        f_in1 = R[j, Li] + L[j, Li_s]
                        f_in2 = L[j + 1, Li]
                        L[j - 1, Li] = self._f(f_in1, f_in2)
                        f_in3 = R[j, Li]
                        f_in4 = L[j + 1, Li]
                        L[j - 1, Li_s] = self._f(f_in3, f_in4) + L[j + 1, Li_s]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        Li = i + k
                        Li_s = i + k + s
                        f_in1 = R[j + 1, Li_s] + L[j + 1, Li_s]
                        f_in2 = R[j, Li]
                        R[j + 1, Li] = self._f(f_in1, f_in2)
                        f_in3 = R[j, Li]
                        f_in4 = L[j + 1, Li]
                        R[j + 1, Li_s] = self._f(f_in3, f_in4) + R[j + 1, Li_s]

            for i in range(N):
                u_hat[i] = 0 if self.frozen_bits[i] else (0 if (L[0, i] + R[0, i]) >= 0 else 1)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            u_hat[i] = 0 if self.frozen_bits[i] else (0 if (L[0, i] + R[0, i]) >= 0 else 1)
        return u_hat.astype(int), num_iters
