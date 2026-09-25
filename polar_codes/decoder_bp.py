"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from channel import hard_decision_llr
from decoder_sc import channel_llr_to_decoder, f_operation
from encoder import polar_encode

LARGE = 1e6


def _f_min_sum(a, b, alpha):
    return alpha * f_operation(a, b)


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        llr_internal = channel_llr_to_decoder(llr_ch)

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)

        L[n, :] = llr_internal
        R[0, :] = 0.0
        R[0, self.frozen_idx] = LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for stage in range(n, 0, -1):
                half = 1 << (stage - 1)
                for i in range(0, N, 2 * half):
                    for j in range(half):
                        left = i + j
                        right = left + half
                        L[stage - 1, left] = _f_min_sum(
                            R[stage, left] + L[stage, right],
                            L[stage, left],
                            self.alpha,
                        )
                        L[stage - 1, right] = _f_min_sum(
                            R[stage, left],
                            L[stage, left],
                            self.alpha,
                        ) + L[stage, right]

            for stage in range(1, n + 1):
                half = 1 << (stage - 1)
                for i in range(0, N, 2 * half):
                    for j in range(half):
                        left = i + j
                        right = left + half
                        R[stage, left] = _f_min_sum(
                            R[stage, right] + L[stage, right],
                            R[stage - 1, left],
                            self.alpha,
                        )
                        R[stage, right] = _f_min_sum(
                            R[stage - 1, left],
                            L[stage, left],
                            self.alpha,
                        ) + R[stage, right]

            total = L[0, :] + R[0, :]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_decision_llr(llr_ch)):
                num_iters = it
                break

        total = L[0, :] + R[0, :]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
