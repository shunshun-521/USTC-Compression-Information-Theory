"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode
from decoder_sc import f_operation, _prepare_llr


def _f_min_sum(x, y, alpha):
    return alpha * f_operation(x, y)


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        """主译码函数。"""
        N = self.N
        n = self.n
        llr = _prepare_llr(llr_ch, N)

        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))
        L[:, 0] = llr
        R[self.frozen_bits, :] = self.LARGE
        R[~self.frozen_bits, :] = 0.0

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for s in range(n):
                block = 1 << (s + 1)
                half = block >> 1
                for j in range(0, N, block):
                    for k in range(half):
                        top = j + k
                        bot = j + half + k
                        L[top, s + 1] = _f_min_sum(
                            L[top, s] + R[top, s], L[bot, s] + R[bot, s], self.alpha
                        )
                        L[bot, s + 1] = _f_min_sum(
                            L[top, s] + R[top, s], L[bot, s], self.alpha
                        ) + L[bot, s] + R[bot, s]

            for s in range(n, 0, -1):
                block = 1 << s
                half = block >> 1
                for j in range(0, N, block):
                    for k in range(half):
                        top = j + k
                        bot = j + half + k
                        R[top, s - 1] = _f_min_sum(
                            R[bot, s] + L[bot, s - 1], R[top, s] + L[top, s - 1], self.alpha
                        )
                        R[bot, s - 1] = _f_min_sum(
                            R[top, s] + L[top, s - 1], L[bot, s - 1], self.alpha
                        ) + R[bot, s] + L[bot, s - 1]

            num_iters = it
            for i in range(N):
                u_hat[i] = 0 if self.frozen_bits[i] or (L[i, n] + R[i, n]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            u_hat[i] = 0 if self.frozen_bits[i] or (L[i, n] + R[i, n]) >= 0 else 1

        return u_hat, num_iters
