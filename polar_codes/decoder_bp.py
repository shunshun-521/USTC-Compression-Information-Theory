"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode
from decoder_sc import _prepare_llr


class BPDecoder:
    """BP 译码器（因子图列 0..n）。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    @staticmethod
    def _f_ms(a, b, alpha):
        sa = np.sign(a)
        sb = np.sign(b)
        sa = np.where(sa == 0, 1.0, sa)
        sb = np.where(sb == 0, 1.0, sb)
        return alpha * sa * sb * np.minimum(np.abs(a), np.abs(b))

    def decode(self, llr_ch):
        llr_raw = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = _prepare_llr(llr_raw)
        n = self.n
        N = self.N
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        hard_x = (llr_raw < 0).astype(int)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for s in range(n - 1, -1, -1):
                step = 1 << s
                for i in range(0, N, step << 1):
                    L[i, s] = self._f_ms(
                        R[i, s + 1] + L[i + step, s + 1],
                        L[i, s + 1],
                        alpha,
                    )
                    L[i + step, s] = (
                        self._f_ms(R[i, s + 1], L[i, s + 1], alpha) + L[i + step, s + 1]
                    )

            for s in range(0, n):
                step = 1 << s
                for i in range(0, N, step << 1):
                    R[i, s + 1] = self._f_ms(
                        R[i + step, s] + L[i + step, s + 1],
                        R[i, s],
                        alpha,
                    )
                    R[i + step, s + 1] = (
                        self._f_ms(R[i, s], L[i, s + 1], alpha) + R[i + step, s]
                    )

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            if np.array_equal(polar_encode(u_hat), hard_x):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
