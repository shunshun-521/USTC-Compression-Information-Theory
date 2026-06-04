"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation_minsum as f_ms


class BPDecoder:
    """BP 译码器（min-sum，含早停）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.LARGE = 1e6

    def _f_ms(self, x, y):
        return self.alpha * f_ms(x, y)

    def decode(self, llr_ch):
        """BP 译码，返回 (u_hat, num_iters)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter
        for it in range(self.max_iter):
            # 右到左更新 L
            for j in range(n, 0, -1):
                s = 2 ** (j - 1)
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        idx = i + t
                        L[idx, j - 1] = self._f_ms(
                            R[idx, j] + L[idx + s, j + 1], L[idx, j + 1]
                        )
                        L[idx + s, j - 1] = self._f_ms(
                            R[idx, j], L[idx, j + 1]
                        ) + L[idx + s, j + 1]

            # 左到右更新 R
            for j in range(1, n + 1):
                s = 2 ** (j - 1)
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        idx = i + t
                        R[idx, j] = self._f_ms(
                            R[idx + s, j] + L[idx + s, j + 1], R[idx, j - 1]
                        )
                        R[idx + s, j] = self._f_ms(
                            R[idx, j - 1], L[idx, j + 1]
                        ) + R[idx + s, j]

            # 早停
            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it + 1
                break

        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
