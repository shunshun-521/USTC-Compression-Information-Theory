"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import _prepare_channel_llrs


def _minsum_f(a, b, alpha=0.9375):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        llr_ch = _prepare_channel_llrs(llr_ch)
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        hard_ch_tree = (llr_ch < 0).astype(int)
        br = bit_reversal_permutation(N)
        hard_x = np.zeros(N, dtype=int)
        hard_x[br] = hard_ch_tree
        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        L[idx, j - 1] = _minsum_f(
                            R[idx, j - 1] + L[idx + s, j],
                            L[idx, j],
                            self.alpha,
                        )
                        L[idx + s, j - 1] = _minsum_f(
                            R[idx, j - 1],
                            L[idx, j],
                            self.alpha,
                        ) + L[idx + s, j]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        R[idx, j + 1] = _minsum_f(
                            R[idx + s, j + 1] + L[idx + s, j + 1],
                            R[idx, j],
                            self.alpha,
                        )
                        R[idx + s, j + 1] = _minsum_f(
                            R[idx, j],
                            L[idx, j + 1],
                            self.alpha,
                        ) + R[idx + s, j + 1]

            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(int)
            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break
        else:
            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(int)

        return u_hat, num_iters
