"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from channel import hard_decision_llr
from decoder_sc import _prepare_channel_llr
from encoder import polar_encode


def _minsum(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


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
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = _prepare_channel_llr(llr_ch, N)
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    L[i, j] = _minsum(
                        R[i, j] + L[i + s, j + 1], L[i, j + 1], alpha
                    )
                    L[i + s, j] = (
                        _minsum(R[i, j], L[i, j + 1], alpha) + L[i + s, j + 1]
                    )

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    R[i, j + 1] = _minsum(
                        R[i + s, j + 1] + L[i + s, j + 1], R[i, j], alpha
                    )
                    R[i + s, j + 1] = (
                        _minsum(R[i, j], L[i, j + 1], alpha) + R[i + s, j]
                    )

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat.astype(int), num_iters
