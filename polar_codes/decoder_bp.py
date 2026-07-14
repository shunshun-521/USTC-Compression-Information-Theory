"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _f_ms(a, b, alpha):
    a = np.clip(a, -19.3, 19.3)
    b = np.clip(b, -19.3, 19.3)
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（向量化 flooding 调度）"""

    LLR_MAX = 19.3

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._stages = self._build_stage_indices(N, self.n)

    @staticmethod
    def _build_stage_indices(N, n):
        stages = []
        for j in range(n):
            s = 1 << j
            i0 = np.arange(0, N, 2 * s)[:, None] + np.arange(s)
            i1 = i0 + s
            stages.append((i0, i1))
        return stages

    def decode(self, llr_ch):
        llr_ch = np.clip(np.asarray(llr_ch, dtype=np.float64), -self.LLR_MAX, self.LLR_MAX)
        N = self.N
        n = self.n
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LLR_MAX
        R[:, n] = 0.0

        x_hard = (llr_ch < 0).astype(int)
        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(n - 1, -1, -1):
                i0, i1 = self._stages[j]
                L[i0, j] = _f_ms(R[i0, j + 1] + L[i1, j + 1], L[i0, j + 1], alpha)
                L[i1, j] = _f_ms(R[i0, j + 1], L[i0, j + 1], alpha) + L[i1, j + 1]

            for j in range(n):
                i0, i1 = self._stages[j]
                R[i0, j + 1] = _f_ms(R[i1, j] + L[i1, j + 1], R[i0, j], alpha)
                R[i1, j + 1] = _f_ms(R[i0, j], L[i0, j + 1], alpha) + R[i1, j]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            if np.array_equal(polar_encode(u_hat), x_hard):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat.astype(int), num_iters
