"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode, bit_reversal_permutation
from channel import hard_decision_llr


def _boxplus_minsum(a, b, alpha=0.9375):
    """min-sum box-plus with normalization"""
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self._large = 1e6

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 u_hat, num_iters
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        alpha = self.alpha

        channel_llr = llr_ch[self.br]

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)

        L[n, :] = channel_llr
        R[0, :] = 0.0
        R[0, self.frozen_bits] = self._large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx1 = i + k
                        idx2 = i + k + step
                        L[j - 1, idx1] = _boxplus_minsum(
                            R[j, idx1] + L[j, idx2], L[j, idx1], alpha
                        )
                        L[j - 1, idx2] = (
                            _boxplus_minsum(R[j, idx1], L[j, idx1], alpha)
                            + L[j, idx2]
                        )

            for j in range(1, n + 1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx1 = i + k
                        idx2 = i + k + step
                        R[j, idx1] = (
                            _boxplus_minsum(R[j - 1, idx1], L[j, idx2], alpha)
                            + R[j, idx2]
                        )
                        R[j, idx2] = _boxplus_minsum(
                            R[j - 1, idx1] + L[j, idx1], R[j, idx2], alpha
                        )

            for i in range(N):
                total = L[0, i] + R[0, i]
                u_hat[i] = 0 if total >= 0 else 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                return u_hat, num_iters
            num_iters = it

        for i in range(N):
            total = L[0, i] + R[0, i]
            u_hat[i] = 0 if total >= 0 else 1
        u_hat[self.frozen_bits] = 0

        return u_hat, num_iters
