"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math

from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _hard_decision_llr(self, llr_ch):
        return (llr_ch < 0).astype(int)

    def _check_early_stop(self, L, R, llr_ch):
        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(self.N, dtype=int)
        u_hat[total < 0] = 1
        u_hat[self.frozen_bits] = 0

        x_hat = polar_encode(u_hat)
        x_hard = self._hard_decision_llr(llr_ch)
        return np.array_equal(x_hat, x_hard), u_hat

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        idx2 = idx + s
                        L[idx, j - 1] = _f_min_sum(
                            R[idx, j - 1] + L[idx2, j],
                            L[idx, j],
                            self.alpha,
                        )
                        L[idx2, j - 1] = _f_min_sum(
                            R[idx, j - 1],
                            L[idx, j],
                            self.alpha,
                        ) + L[idx2, j]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        idx2 = idx + s
                        R[idx, j + 1] = _f_min_sum(
                            R[idx2, j] + L[idx2, j + 1],
                            R[idx, j],
                            self.alpha,
                        )
                        R[idx2, j + 1] = _f_min_sum(
                            R[idx, j],
                            L[idx, j + 1],
                            self.alpha,
                        ) + R[idx2, j]

            stopped, u_hat = self._check_early_stop(L, R, llr_ch)
            if stopped:
                break

        if num_iters == self.max_iter:
            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[total < 0] = 1
            u_hat[self.frozen_bits] = 0

        return u_hat, num_iters
