"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import _prepare_llr
from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """
    BP 译码器。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._LARGE = 1e6

    def decode(self, llr_ch):
        """
        主译码函数。返回 (u_hat, num_iters)
        """
        llr_ch = _prepare_llr(np.asarray(llr_ch, dtype=np.float64))
        N, n = self.N, self.n
        alpha = self.alpha

        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self._LARGE

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    L[i, j - 1] = _f_min_sum(
                        R[i, j] + L[i + s, j], L[i, j], alpha
                    )
                    L[i + s, j - 1] = _f_min_sum(R[i, j], L[i, j], alpha) + L[i + s, j]

            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    R[i, j] = _f_min_sum(
                        R[i + s, j] + L[i + s, j], R[i, j - 1], alpha
                    )
                    R[i + s, j] = (
                        _f_min_sum(R[i, j - 1], L[i, j], alpha) + R[i + s, j]
                    )

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            rev_hard = hard_ch  # llr_ch already prepared
            if np.array_equal(x_hat, rev_hard):
                num_iters = it
                break
        else:
            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

        return u_hat, num_iters
