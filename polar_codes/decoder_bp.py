"""
极化码 BP（置信传播）译码器
基于因子图的 min-sum 近似与 SC 先验初始化，含早停
"""
import math

import numpy as np

from channel import hard_decision_llr
from decoder_sc import sc_decode
from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器：SC 初始化 + 因子图 min-sum 迭代细化"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _one_pass(self, L, R, llr_ch):
        n, N, alpha = self.n, self.N, self.alpha
        L_old = L.copy()
        R_old = R.copy()
        R_old[:, 0] = R[:, 0]

        for j in range(n - 1, -1, -1):
            step = 1 << j
            for i in range(0, N, 2 * step):
                L[i, j] = _f_min_sum(
                    R_old[i, j + 1] + L_old[i + step, j + 1],
                    L_old[i, j + 1],
                    alpha,
                )
                L[i + step, j] = _f_min_sum(
                    R_old[i, j + 1], L_old[i, j + 1], alpha
                ) + L_old[i + step, j + 1]

        for j in range(0, n):
            step = 1 << j
            for i in range(0, N, 2 * step):
                R[i, j + 1] = _f_min_sum(
                    R_old[i + step, j + 1] + L_old[i + step, j + 1],
                    R_old[i, j],
                    alpha,
                )
                R[i + step, j + 1] = _f_min_sum(
                    R_old[i, j], L_old[i, j + 1], alpha
                ) + R_old[i + step, j]

        L[:, n] = llr_ch

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch

        u_hat = sc_decode(llr_ch, self.frozen_bits)
        for i in range(N):
            if self.frozen_bits[i]:
                R[i, 0] = self.large
            else:
                R[i, 0] = (1 - 2 * u_hat[i]) * 10.0

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it
            self._one_pass(L, R, llr_ch)

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_decision_llr(llr_ch)):
                break

            # 软信息反馈（阻尼）以继续迭代
            extrinsic = L[:, 0] + R[:, 0]
            llr_ch = 0.75 * llr_ch + 0.25 * extrinsic

        return u_hat, num_iters
