"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode, prepare_decoder_llr


class BPDecoder:
    """BP 译码器（因子图 min-sum）。"""

    LARGE = 1e10

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

    def _hard_bits(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat

    def decode(self, llr_ch):
        llr_ch = prepare_decoder_llr(llr_ch)
        n, N = self.n, self.N

        # L[i][j], R[i][j]: i 为行（0..N-1），j 为层（0..n）
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it

            # 右到左：更新 L 消息
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for block in range(0, N, 2 * step):
                    for k in range(step):
                        i = block + k
                        ip = i + step
                        L[i, j - 1] = self._f_ms(
                            R[i, j] + L[ip, j], L[i, j]
                        )
                        L[ip, j - 1] = self._f_ms(R[i, j], L[i, j]) + L[ip, j]

            # 左到右：更新 R 消息
            for j in range(0, n):
                step = 1 << j
                for block in range(0, N, 2 * step):
                    for k in range(step):
                        i = block + k
                        ip = i + step
                        R[i, j + 1] = self._f_ms(
                            R[ip, j] + L[ip, j + 1], R[i, j]
                        )
                        R[ip, j + 1] = self._f_ms(R[i, j], L[i, j + 1]) + R[ip, j]

            u_hat = self._hard_bits(L, R)
            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                break

        u_hat = self._hard_bits(L, R)
        return u_hat, num_iters
