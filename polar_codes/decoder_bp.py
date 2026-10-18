"""
极化码 BP（置信传播）译码器
min-sum 近似 + 早停
"""
import numpy as np
import math
from encoder import polar_encode


def _ms_f(a, b, alpha):
    sa = 1.0 if a >= 0 else -1.0
    sb = 1.0 if b >= 0 else -1.0
    return alpha * sa * sb * min(abs(a), abs(b))


class BPDecoder:
    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.LARGE = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        # L[i][j], R[i][j]：i 为行（0..N-1），j 为列（0..n）
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        ms = self.alpha
        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            # 右到左更新 L
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    L[i, j - 1] = _ms_f(
                        R[i, j] + L[i + s, j], L[i, j], ms
                    )
                    L[i + s, j - 1] = _ms_f(R[i, j], L[i, j], ms) + L[i + s, j]

            # 左到右更新 R
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    R[i, j + 1] = _ms_f(
                        R[i + s, j] + L[i + s, j + 1], R[i, j], ms
                    )
                    R[i + s, j + 1] = _ms_f(R[i, j], L[i, j + 1], ms) + R[i + s, j]

            # 早停
            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
