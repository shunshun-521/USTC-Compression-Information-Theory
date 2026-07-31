"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import _prepare_channel_llrs
from encoder import polar_encode


def _minsum_f(x, y, alpha):
    """min-sum 近似 f 运算，带修正因子 alpha。"""
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    """

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def _hard_decision_llr(self, llr_ch):
        return (llr_ch < 0).astype(int)

    def _get_u_hat(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total[i] >= 0 else 1
        return u_hat

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        N = self.N
        n = self.n
        alpha = self.alpha
        llr_ch = _prepare_channel_llrs(llr_ch, N)

        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 2 ** (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        L[idx, j - 1] = _minsum_f(
                            R[idx, j - 1] + L[idx + s, j],
                            L[idx, j],
                            alpha,
                        )
                        L[idx + s, j - 1] = _minsum_f(
                            R[idx, j - 1],
                            L[idx, j],
                            alpha,
                        ) + L[idx + s, j]

            for j in range(0, n):
                s = 2 ** j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        R[idx, j + 1] = _minsum_f(
                            R[idx + s, j] + L[idx + s, j + 1],
                            R[idx, j],
                            alpha,
                        )
                        R[idx + s, j + 1] = _minsum_f(
                            R[idx, j],
                            L[idx + s, j + 1],
                            alpha,
                        ) + R[idx + s, j]

            u_hat = self._get_u_hat(L, R)
            x_hat = polar_encode(u_hat)
            hard_ch = self._hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        u_hat = self._get_u_hat(L, R)
        return u_hat, num_iters
