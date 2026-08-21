"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import _align_llr_for_decoder, upper_llr
from encoder import bit_reversal_permutation, polar_encode


def _boxplus(x, y, alpha=1.0):
    if alpha >= 0.999:
        return upper_llr(x, y)
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e7

    def decode(self, llr_ch):
        llr_raw = np.asarray(llr_ch, dtype=np.float64)
        llr = _align_llr_for_decoder(llr_raw)
        n = self.n
        N = self.N

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr
        R[0, :] = 0.0
        R[0, self.frozen_bits] = self.large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)
        br = bit_reversal_permutation(N)

        for it in range(1, self.max_iter + 1):
            for layer in range(n, 0, -1):
                step = 1 << (layer - 1)
                for block in range(0, N, step << 1):
                    base = block
                    left = slice(base, base + step)
                    right = slice(base + step, base + 2 * step)
                    L[layer - 1, left] = _boxplus(
                        R[layer, left] + L[layer, right],
                        L[layer, left],
                        self.alpha,
                    )
                    L[layer - 1, right] = _boxplus(R[layer, left], L[layer, left], self.alpha) + L[
                        layer, right
                    ]

            for layer in range(0, n):
                step = 1 << layer
                for block in range(0, N, step << 1):
                    base = block
                    left = slice(base, base + step)
                    right = slice(base + step, base + 2 * step)
                    R[layer + 1, left] = _boxplus(
                        R[layer + 1, right] + L[layer + 1, right],
                        R[layer, left],
                        self.alpha,
                    )
                    R[layer + 1, right] = _boxplus(R[layer, left], L[layer + 1, left], self.alpha) + R[
                        layer + 1, right
                    ]

            post = L[0, :] + R[0, :]
            u_hat = (post < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            x_hard = np.zeros(N, dtype=int)
            x_hard[br] = (llr < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

        u_out = u_hat.copy()
        return u_out, num_iters
