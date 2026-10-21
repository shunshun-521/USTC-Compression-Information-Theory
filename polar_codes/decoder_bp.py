"""
极化码 BP（置信传播）译码器
因子图 min-sum，含早停
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（列 0..n，行 0..N-1）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.max_iter = int(max_iter)
        self.alpha = float(alpha)
        frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_mask = frozen_bits.astype(bool)
        self.large = 1e6

    def _f(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_mask, 0] = self.large

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it
            # 右到左更新 L
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        i0 = i + k
                        i1 = i + k + s
                        L[i0, j - 1] = self._f(
                            R[i0, j] + L[i1, j], L[i0, j]
                        )
                        L[i1, j - 1] = self._f(R[i0, j], L[i0, j]) + L[i1, j]

            # 左到右更新 R
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        i0 = i + k
                        i1 = i + k + s
                        R[i0, j + 1] = self._f(
                            R[i1, j] + L[i1, j + 1], R[i0, j]
                        )
                        R[i1, j + 1] = self._f(R[i0, j], L[i0, j + 1]) + R[i1, j]

            u_hat = np.zeros(N, dtype=np.int8)
            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(np.int8)
            u_hat[self.frozen_mask] = 0

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard_x):
                break

        u_hat = np.zeros(N, dtype=np.int8)
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(np.int8)
        u_hat[self.frozen_mask] = 0
        return u_hat, num_iters
