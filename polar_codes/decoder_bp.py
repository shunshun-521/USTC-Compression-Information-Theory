"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _f_min_sum(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """
    BP 译码器。
    """

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.m = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        """
        主译码函数。
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.N
        m = self.m

        l_msg = np.zeros((n, m + 1), dtype=np.float64)
        r_msg = np.zeros((n, m + 1), dtype=np.float64)

        l_msg[:, m] = llr_ch
        r_msg[:, 0] = 0.0
        r_msg[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(n, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(m, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, n, 2 * s):
                    for t in range(s):
                        i0 = i + t
                        i1 = i + t + s
                        l_msg[i0, j - 1] = _f_min_sum(
                            r_msg[i0, j] + l_msg[i1, j],
                            l_msg[i0, j],
                            self.alpha,
                        )
                        l_msg[i1, j - 1] = _f_min_sum(
                            r_msg[i0, j],
                            l_msg[i0, j],
                            self.alpha,
                        ) + l_msg[i1, j]

            for j in range(0, m):
                s = 1 << j
                for i in range(0, n, 2 * s):
                    for t in range(s):
                        i0 = i + t
                        i1 = i + t + s
                        r_msg[i0, j + 1] = _f_min_sum(
                            r_msg[i1, j] + l_msg[i1, j + 1],
                            r_msg[i0, j],
                            self.alpha,
                        )
                        r_msg[i1, j + 1] = _f_min_sum(
                            r_msg[i0, j],
                            l_msg[i0, j + 1],
                            self.alpha,
                        ) + r_msg[i1, j]

            for i in range(n):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (l_msg[i, 0] + r_msg[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                num_iters = it
                break

        for i in range(n):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (l_msg[i, 0] + r_msg[i, 0]) >= 0 else 1

        return u_hat, num_iters
