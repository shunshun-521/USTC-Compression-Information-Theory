"""
极化码 BP（置信传播）译码器
基于因子图，min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        if 2 ** self.n != N:
            raise ValueError("N must be a power of 2")
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        # L[i][j], R[i][j]: i in 0..N-1, j in 0..n
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self._large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            # 右 -> 左 更新 L
            for j in range(n, 0, -1):
                s = 2 ** (j - 1)
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        i0 = i + t
                        i1 = i + t + s
                        L[i0, j - 1] = self._f_ms(
                            R[i0, j] + L[i1, j + 1], L[i0, j + 1]
                        )
                        L[i1, j - 1] = self._f_ms(R[i0, j], L[i0, j + 1]) + L[i1, j + 1]

            # 左 -> 右 更新 R
            for j in range(1, n + 1):
                s = 2 ** (j - 1)
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        i0 = i + t
                        i1 = i + t + s
                        R[i0, j] = self._f_ms(
                            R[i1, j] + L[i1, j + 1], R[i0, j - 1]
                        )
                        R[i1, j] = self._f_ms(R[i0, j - 1], L[i0, j + 1]) + R[i1, j]

            # 早停
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
