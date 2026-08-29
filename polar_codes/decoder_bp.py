"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6

    def _g(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self._large

        num_iters = self.max_iter

        for it in range(self.max_iter):
            for stage in range(n - 1, -1, -1):
                step = 1 << stage
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        a = i + k
                        b = a + step
                        L[a, stage] = self._g(
                            L[a, stage + 1], R[b, stage] + L[b, stage + 1]
                        )
                        L[b, stage] = self._g(
                            L[a, stage + 1], R[a, stage]
                        ) + L[b, stage + 1]

            for stage in range(n):
                step = 1 << stage
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        a = i + k
                        b = a + step
                        R[a, stage + 1] = self._g(
                            R[a, stage], R[b, stage] + L[b, stage + 1]
                        )
                        R[b, stage + 1] = self._g(
                            R[a, stage], L[a, stage + 1]
                        ) + R[b, stage]

            u_hat = np.zeros(N, dtype=np.int8)
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it + 1
                break

        u_hat = np.zeros(N, dtype=np.int8)
        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters

