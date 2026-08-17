"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import _prepare_channel_llr


def _f_min_sum(x, y, alpha):
    """min-sum f 运算，带修正因子 alpha"""
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_indices = np.where(self.frozen_bits == 1)[0]
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        llr_ch = _prepare_channel_llr(llr_ch)
        N, n = self.N, self.n
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_indices, 0] = self.LARGE

        num_iters = 0

        for it in range(self.max_iter):
            num_iters = it + 1

            for j in range(n - 1, -1, -1):
                s = 2 ** j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        L[idx, j] = _f_min_sum(
                            R[idx, j] + L[idx + s, j + 1],
                            L[idx, j + 1],
                            alpha,
                        )
                        L[idx + s, j] = (
                            _f_min_sum(R[idx, j], L[idx, j + 1], alpha)
                            + L[idx + s, j + 1]
                        )

            for j in range(n - 1):
                s = 2 ** j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        R[idx, j + 1] = _f_min_sum(
                            R[idx + s, j] + L[idx + s, j + 1],
                            R[idx, j],
                            alpha,
                        )
                        R[idx + s, j + 1] = (
                            _f_min_sum(R[idx, j], L[idx, j + 1], alpha)
                            + R[idx + s, j]
                        )

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_indices] = 0

            x_hat = polar_encode(u_hat)
            br = bit_reversal_permutation(N)
            hard_ch = (llr_ch[br] < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_indices] = 0

        return u_hat, num_iters
