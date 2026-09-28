"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import f_operation, g_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器（列索引 0..n，行 0..N-1）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.max_iter = max_iter
        self.alpha = alpha
        fb = np.asarray(frozen_bits).astype(bool)
        self.frozen_set = set(np.nonzero(fb)[0])
        self.large = 1e6

    def _f_ms(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        for idx in self.frozen_set:
            R[idx, 0] = self.large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            # 右到左更新 L
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, step << 1):
                    for t in range(step):
                        L[i + t, j - 1] = self._f_ms(
                            R[i + t, j - 1] + L[i + t, j],
                            L[i + t + step, j],
                        )
                        L[i + t + step, j - 1] = self._f_ms(
                            R[i + t, j - 1],
                            L[i + t, j],
                        ) + L[i + t + step, j]

            # 左到右更新 R
            for j in range(1, n + 1):
                step = 1 << (j - 1)
                for i in range(0, N, step << 1):
                    for t in range(step):
                        R[i + t, j] = self._f_ms(
                            R[i + t + step, j - 1] + L[i + t + step, j],
                            R[i + t, j - 1],
                        )
                        R[i + t + step, j - 1] = self._f_ms(
                            R[i + t, j - 1],
                            L[i + t, j],
                        ) + R[i + t + step, j]

            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                if i in self.frozen_set:
                    u_hat[i] = 0
                else:
                    total = L[i, 0] + R[i, 0]
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            if i in self.frozen_set:
                u_hat[i] = 0
            else:
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1
        return u_hat, num_iters
