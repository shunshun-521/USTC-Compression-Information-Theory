"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode
from channel import hard_decision_llr


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits.astype(bool))[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_idx] = self.LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for layer in range(n - 1, -1, -1):
                step = 1 << layer
                for i in range(0, N, 2 * step):
                    la = L[layer + 1, i:i + step]
                    lb = L[layer + 1, i + step:i + 2 * step]
                    ra = R[layer, i:i + step]
                    L[layer, i:i + step] = _f_min_sum(
                        ra + lb, la, self.alpha
                    )
                    L[layer, i + step:i + 2 * step] = _f_min_sum(
                        ra, la, self.alpha
                    ) + lb

            for layer in range(0, n):
                step = 1 << layer
                for i in range(0, N, 2 * step):
                    la = L[layer + 1, i:i + step]
                    lb = L[layer + 1, i + step:i + 2 * step]
                    rb = R[layer + 1, i + step:i + 2 * step]
                    ra = R[layer, i:i + step]
                    R[layer + 1, i:i + step] = _f_min_sum(
                        rb + lb, ra, self.alpha
                    )
                    R[layer + 1, i + step:i + 2 * step] = _f_min_sum(
                        ra, la, self.alpha
                    ) + rb

            u_hat = self._hard_decision(L, R)
            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_decision_llr(llr_ch)):
                num_iters = it
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters

    def _hard_decision(self, L, R):
        total = L[0, :] + R[0, :]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits.astype(bool)] = 0
        return u_hat
