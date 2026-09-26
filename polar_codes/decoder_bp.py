"""
极化码 BP（置信传播）译码器
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
        self.LARGE = 1e6

    def _f_ms(self, x, y):
        sx = np.sign(x)
        sy = np.sign(y)
        sx = np.where(sx == 0, 1.0, sx)
        sy = np.where(sy == 0, 1.0, sy)
        return self.alpha * sx * sy * np.minimum(np.abs(x), np.abs(y))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        hard_x = (llr_ch < 0).astype(int)
        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for t in range(step):
                        idx = i + t
                        L[idx, j - 1] = self._f_ms(
                            R[idx, j] + L[idx + step, j + 1], L[idx, j + 1]
                        )
                        L[idx + step, j - 1] = self._f_ms(
                            R[idx, j], L[idx, j + 1]
                        ) + L[idx + step, j + 1]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    for t in range(step):
                        idx = i + t
                        R[idx, j + 1] = self._f_ms(
                            R[idx + step, j] + L[idx + step, j + 1], R[idx, j]
                        )
                        R[idx + step, j + 1] = self._f_ms(
                            R[idx, j], L[idx, j + 1]
                        ) + R[idx + step, j]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0
            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
