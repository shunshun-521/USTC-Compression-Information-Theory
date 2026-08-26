"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from channel import hard_decision_llr
from decoder_sc import f_operation, prepare_decoder_llr
from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    sign = np.sign(a) * np.sign(b)
    sign = np.where(sign == 0, 1.0, sign)
    return alpha * sign * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.large = 1e6

    def decode(self, llr_ch):
        llr_raw = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = prepare_decoder_llr(llr_raw)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    La = R[i, j - 1] + L[i + step, j]
                    Lb = L[i, j]
                    L[i, j - 1] = _f_min_sum(La, Lb, self.alpha)
                    L[i + step, j - 1] = _f_min_sum(R[i, j - 1], L[i, j], self.alpha) + L[i + step, j]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    Ra = R[i + step, j + 1] + L[i + step, j + 1]
                    Rb = R[i, j]
                    R[i, j + 1] = _f_min_sum(Ra, Rb, self.alpha)
                    R[i + step, j + 1] = _f_min_sum(R[i, j], L[i, j + 1], self.alpha) + R[i + step, j]

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0.0 else 1
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_raw)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            u_hat[i] = 0 if total >= 0.0 else 1
        u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
