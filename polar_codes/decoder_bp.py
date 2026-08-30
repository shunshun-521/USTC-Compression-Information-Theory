"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math
from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    sign = np.sign(a) * np.sign(b)
    mag = np.minimum(np.abs(a), np.abs(b))
    return alpha * sign * mag


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
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]
        self.large = 1e6

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        i0 = i + k
                        i1 = i + k + step
                        L[i0, j - 1] = _f_min_sum(
                            R[i0, j] + L[i1, j],
                            L[i0, j],
                            self.alpha,
                        )
                        L[i1, j - 1] = _f_min_sum(
                            R[i0, j],
                            L[i0, j],
                            self.alpha,
                        ) + L[i1, j]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        i0 = i + k
                        i1 = i + k + step
                        R[i0, j + 1] = _f_min_sum(
                            R[i1, j] + L[i1, j + 1],
                            R[i0, j],
                            self.alpha,
                        )
                        R[i1, j + 1] = _f_min_sum(
                            R[i0, j],
                            L[i0, j + 1],
                            self.alpha,
                        ) + R[i1, j]

            num_iters = it
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
