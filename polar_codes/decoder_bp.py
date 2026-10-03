"""
极化码 BP（置信传播）译码器
基于极化因子图，min-sum 近似，含早停
"""
import math
import numpy as np

from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（与当前编码器一致的因子图布局）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        LARGE = 1e8

        # L[s, i]: 从信道向右到比特 i 的 LLR 消息；R[s, i]: 从比特向左的消息
        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_idx] = LARGE

        num_iters = 0
        for it in range(self.max_iter):
            num_iters = it + 1
            for s in range(n - 1, -1, -1):
                step = 1 << s
                for i in range(0, N, 2 * step):
                    for j in range(i, i + step):
                        L[s, j] = self._f_ms(
                            R[s, j] + L[s + 1, j + step], L[s + 1, j]
                        )
                        L[s, j + step] = (
                            self._f_ms(R[s, j], L[s + 1, j]) + L[s + 1, j + step]
                        )

            for s in range(0, n):
                step = 1 << s
                for i in range(0, N, 2 * step):
                    for j in range(i, i + step):
                        R[s + 1, j] = self._f_ms(
                            R[s + 1, j + step] + L[s + 1, j + step], R[s, j]
                        )
                        R[s + 1, j + step] = (
                            self._f_ms(R[s, j], L[s + 1, j]) + R[s + 1, j + step]
                        )

            total = L[0, :] + R[0, :]
            u_hat = np.zeros(N, dtype=int)
            u_hat[self.info_idx] = (total[self.info_idx] < 0).astype(int)
            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                break

        total = L[0, :] + R[0, :]
        u_hat = np.zeros(N, dtype=int)
        u_hat[self.info_idx] = (total[self.info_idx] < 0).astype(int)
        return u_hat, num_iters
