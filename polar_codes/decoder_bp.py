"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import f_operation, _prepare_channel_llr
from encoder import polar_encode


LARGE = 1e6


class BPDecoder:
    """BP 译码器（因子图列 0..n）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = _as_frozen_mask(frozen_bits)
        self.max_iter = max_iter
        self.alpha = alpha

    def _boxplus(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = _prepare_channel_llr(np.asarray(llr_ch, dtype=np.float64))
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen, 0] = LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # 左到右更新 R
            for phase in range(n):
                step = 1 << phase
                for i in range(0, N, 2 * step):
                    for j in range(step):
                        R[i + j, phase + 1] = self._boxplus(
                            R[i + j + step, phase] + L[i + j + step, phase + 1],
                            R[i + j, phase],
                        )
                        R[i + j + step, phase + 1] = self._boxplus(
                            R[i + j, phase], L[i + j, phase + 1]
                        ) + R[i + j + step, phase]

            # 右到左更新 L
            for phase in range(n - 1, -1, -1):
                step = 1 << phase
                for i in range(0, N, 2 * step):
                    for j in range(step):
                        L[i + j, phase] = self._boxplus(
                            R[i + j, phase] + L[i + j + step, phase + 1],
                            L[i + j, phase + 1],
                        )
                        L[i + j + step, phase] = self._boxplus(
                            R[i + j, phase], L[i + j, phase + 1]
                        ) + L[i + j + step, phase + 1]

            num_iters = it

            for i in range(N):
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
            u_hat[self.frozen] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
        u_hat[self.frozen] = 0

        return u_hat.astype(int), num_iters


def _as_frozen_mask(frozen_bits):
    frozen = np.asarray(frozen_bits)
    if frozen.dtype == bool:
        return frozen
    return frozen.astype(bool)
