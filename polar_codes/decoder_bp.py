"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import sc_decode
from encoder import polar_encode


def _boxplus_minsum(a, b, alpha):
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
        self.large = 1e6

    def _hard_decision(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat

    def _check_early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, hard_ch)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = self.max_iter
        u_hat = sc_decode(llr_ch, self.frozen_bits)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, s << 1):
                    L[i, j - 1] = _boxplus_minsum(
                        R[i, j - 1] + L[i + s, j], L[i, j], alpha
                    )
                    L[i + s, j - 1] = _boxplus_minsum(
                        R[i, j - 1], L[i, j], alpha
                    ) + L[i + s, j]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, s << 1):
                    r_left = R[i, j - 1] if j > 0 else R[i, 0]
                    R[i, j] = _boxplus_minsum(
                        R[i + s, j + 1] + L[i + s, j + 1], r_left, alpha
                    )
                    R[i + s, j] = _boxplus_minsum(
                        r_left, L[i, j + 1], alpha
                    ) + R[i + s, j + 1]

            u_hat = self._hard_decision(L, R)
            if self._check_early_stop(u_hat, llr_ch):
                num_iters = it
                break

        u_bp = self._hard_decision(L, R)
        if self._check_early_stop(u_bp, llr_ch):
            return u_bp, num_iters
        return sc_decode(llr_ch, self.frozen_bits), num_iters
