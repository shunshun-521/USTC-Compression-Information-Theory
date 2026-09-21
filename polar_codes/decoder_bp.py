"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode, bit_reversal_permutation


LARGE = 1e6


def _f_min_sum(a, b, alpha):
    """min-sum f 运算，带修正因子 alpha。"""
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（因子图 min-sum，含早停）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 u_hat 和实际迭代次数。
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = LARGE

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it

            for j in range(n, 0, -1):
                s = 2 ** (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        L[idx, j - 1] = _f_min_sum(
                            R[idx, j] + L[idx + s, j],
                            L[idx, j],
                            self.alpha,
                        )
                        L[idx + s, j - 1] = _f_min_sum(
                            R[idx, j], L[idx, j], self.alpha
                        ) + L[idx + s, j]

            for j in range(0, n):
                s = 2 ** (j)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        R[idx, j + 1] = _f_min_sum(
                            R[idx + s, j] + L[idx + s, j + 1],
                            R[idx, j],
                            self.alpha,
                        )
                        R[idx + s, j + 1] = _f_min_sum(
                            R[idx, j], L[idx, j + 1], self.alpha
                        ) + R[idx + s, j]

            u_hat = self._hard_decision(L, R)
            if self._check_early_stop(u_hat, llr_ch):
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters

    def _hard_decision(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat

    def _check_early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, hard_ch)
