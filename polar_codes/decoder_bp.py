"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import _permute_channel_llr
from encoder import bit_reversal_permutation, polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.LARGE = 1e6
        br = bit_reversal_permutation(N)
        self._inv_br = np.zeros(N, dtype=int)
        self._inv_br[br] = np.arange(N)

    def decode(self, llr_ch):
        llr_perm = _permute_channel_llr(np.asarray(llr_ch, dtype=np.float64))
        n, N = self.n, self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_perm
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            for s in range(n - 1, -1, -1):
                step = 1 << s
                for i in range(0, N, 2 * step):
                    L[i, s] = _f_min_sum(
                        R[i, s] + L[i + step, s + 1], L[i, s + 1], self.alpha
                    )
                    L[i + step, s] = _f_min_sum(
                        R[i, s], L[i, s + 1], self.alpha
                    ) + L[i + step, s + 1]

            for s in range(n):
                step = 1 << s
                for i in range(0, N, 2 * step):
                    R[i, s + 1] = _f_min_sum(
                        R[i + step, s + 1] + L[i + step, s + 1], R[i, s], self.alpha
                    )
                    R[i + step, s + 1] = _f_min_sum(
                        R[i, s], L[i, s + 1], self.alpha
                    ) + R[i + step, s]

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat_perm = polar_encode(u_hat)[self._inv_br]
            x_hard = (llr_perm < 0).astype(int)
            if np.array_equal(x_hat_perm, x_hard):
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
