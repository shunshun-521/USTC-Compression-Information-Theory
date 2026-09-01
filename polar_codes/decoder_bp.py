"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from channel import hard_decision_llr
from encoder import polar_encode


def _boxplus_minsum(a, b, alpha):
    sa = 1.0 if a >= 0 else -1.0
    sb = 1.0 if b >= 0 else -1.0
    return alpha * sa * sb * min(abs(a), abs(b))


class BPDecoder:
    """
    BP 译码器。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.info_idx = np.where(self.frozen_bits == 0)[0]

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        LARGE = 1e6

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx = i + k
                        idx2 = idx + step
                        if j == n:
                            L[idx, j - 1] = _boxplus_minsum(
                                R[idx, j] + L[idx2, j],
                                L[idx, j],
                                self.alpha,
                            )
                            L[idx2, j - 1] = _boxplus_minsum(
                                R[idx, j],
                                L[idx, j],
                                self.alpha,
                            ) + L[idx2, j]
                        else:
                            L[idx, j - 1] = _boxplus_minsum(
                                R[idx, j] + L[idx2, j + 1],
                                L[idx, j + 1],
                                self.alpha,
                            )
                            L[idx2, j - 1] = _boxplus_minsum(
                                R[idx, j],
                                L[idx, j + 1],
                                self.alpha,
                            ) + L[idx2, j + 1]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx = i + k
                        idx2 = idx + step
                        R[idx, j + 1] = _boxplus_minsum(
                            R[idx2, j] + L[idx2, j + 1],
                            R[idx, j],
                            self.alpha,
                        )
                        R[idx2, j + 1] = _boxplus_minsum(
                            R[idx, j],
                            L[idx, j + 1],
                            self.alpha,
                        ) + R[idx2, j]

            for i in range(N):
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_decision_llr(llr_ch)):
                num_iters = it
                break

        for i in range(N):
            u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
        u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
