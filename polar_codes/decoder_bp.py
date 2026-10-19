"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（分层 up/down 消息，min-sum）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]

    def _f_ms(self, a, b):
        if self.alpha >= 0.99:
            return f_operation(a, b)
        sa = np.where(a >= 0, 1.0, -1.0)
        sb = np.where(b >= 0, 1.0, -1.0)
        return self.alpha * sa * sb * np.minimum(np.abs(a), np.abs(b))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        LARGE = 1e6

        up = np.zeros((n + 1, N), dtype=np.float64)
        down = np.zeros((n + 1, N), dtype=np.float64)
        up[n] = llr_ch

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it
            up.fill(0.0)
            up[n] = llr_ch
            down.fill(0.0)
            down[0, self.frozen_idx] = LARGE

            for stage in range(n - 1, -1, -1):
                m = 1 << stage
                for i in range(0, N, 2 * m):
                    for j in range(m):
                        a = i + j
                        b = i + j + m
                        up[stage, a] = self._f_ms(
                            up[stage + 1, a] + down[stage + 1, b], up[stage + 1, b]
                        )
                        up[stage, b] = (
                            self._f_ms(down[stage + 1, a], up[stage + 1, b])
                            + up[stage + 1, b]
                        )

            for stage in range(n):
                m = 1 << stage
                for i in range(0, N, 2 * m):
                    for j in range(m):
                        a = i + j
                        b = i + j + m
                        down[stage + 1, b] = self._f_ms(
                            down[stage + 1, a] + up[stage + 1, b], down[stage, a]
                        )
                        down[stage + 1, a] = self._f_ms(down[stage, a], up[stage + 1, b])

            total = up[0] + down[0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[self.info_idx] = (total[self.info_idx] < 0).astype(int)
            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                return u_hat, num_iters

        total = up[0] + down[0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[self.info_idx] = (total[self.info_idx] < 0).astype(int)
        return u_hat, num_iters
