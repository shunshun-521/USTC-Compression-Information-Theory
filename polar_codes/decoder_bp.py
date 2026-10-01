"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（分层因子图）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_idx = set(np.where(self.frozen_bits)[0])
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_ms(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        # L[i][j], R[i][j]: i=行(比特位置), j=列(层)
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch

        for idx in self.frozen_idx:
            R[idx, 0] = self.LARGE

        num_iters = 0
        for it in range(self.max_iter):
            num_iters = it + 1

            # 从右到左更新 L
            for j in range(n, 0, -1):
                s = 2 ** (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        top = i + k
                        bot = i + k + s
                        L[top, j - 1] = self._f_ms(
                            R[top, j] + L[bot, j], L[top, j]
                        )
                        L[bot, j - 1] = self._f_ms(R[top, j], L[top, j]) + L[bot, j]

            # 从左到右更新 R
            for j in range(1, n + 1):
                s = 2 ** (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        top = i + k
                        bot = i + k + s
                        R[top, j] = self._f_ms(
                            R[bot, j] + L[bot, j], R[top, j - 1]
                        )
                        R[bot, j] = self._f_ms(R[top, j - 1], L[top, j]) + R[bot, j]

            # 早停检查
            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                if i in self.frozen_idx:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            if i in self.frozen_idx:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
