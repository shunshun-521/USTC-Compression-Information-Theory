"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math

from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation, g_operation, _frozen_bool

_LARGE = 1e6


class BPDecoder:
    """BP 译码器（因子图列 0..n）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = _frozen_bool(frozen_bits)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        # L[i][j]: 自右向左；R[i][j]: 自左向右
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch[self.br]

        R[:, 0] = 0.0
        R[self.frozen, 0] = _LARGE

        for it in range(self.max_iter):
            # 右到左 L
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, step << 1):
                    La = R[i, j] + L[i, j]
                    Lb = L[i + step, j]
                    L[i, j - 1] = self._f_ms(La, Lb)
                    L[i + step, j - 1] = self._f_ms(R[i, j], L[i, j]) + L[i + step, j]

            # 左到右 R
            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, step << 1):
                    R[i, j + 1] = self._f_ms(R[i + step, j] + L[i + step, j + 1], R[i, j])
                    R[i + step, j + 1] = self._f_ms(R[i, j], L[i, j + 1]) + R[i + step, j]

            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                if self.frozen[i]:
                    u_hat[i] = 0
                else:
                    s = L[i, 0] + R[i, 0]
                    u_hat[i] = 0 if s >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                return u_hat, it + 1

        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            if self.frozen[i]:
                u_hat[i] = 0
            else:
                s = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if s >= 0 else 1
        return u_hat, self.max_iter
