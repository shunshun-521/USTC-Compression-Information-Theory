"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
参考 Abbas et al. 的 BP 更新公式 (min-sum)
"""
import math

import numpy as np

from channel import hard_decision_llr
from encoder import bit_reversal_permutation, polar_encode


def _ms(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """
    BP 译码器。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.m = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.rev = bit_reversal_permutation(N)
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]
        self.large = 1e7

    def _decide(self, L, R):
        u_hat = np.zeros(self.N, dtype=np.int8)
        belief = L[:, 0] + R[:, 0]
        u_hat[self.info_idx] = (belief[self.info_idx] < 0).astype(np.int8)
        return u_hat

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 u_hat, num_iters
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        channel = llr_ch[self.rev]

        L = np.zeros((self.N, self.m + 1), dtype=np.float64)
        R = np.zeros((self.N, self.m + 1), dtype=np.float64)
        L[:, self.m] = channel
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.large

        num_iters = 0
        u_hat = np.zeros(self.N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            Ln = L.copy()
            Rn = R.copy()

            for j_paper in range(1, self.m + 1):
                j = j_paper - 1
                step = 1 << (self.m - j_paper)
                for i in range(0, self.N, 2 * step):
                    for k in range(step):
                        top = i + k
                        bot = i + k + step
                        Ln[bot, j] = L[bot, j + 1] + _ms(
                            L[top, j + 1], R[top, j], self.alpha
                        )
                        Ln[top, j] = _ms(
                            L[top, j + 1],
                            L[bot, j + 1] + R[bot, j],
                            self.alpha,
                        )
                        Rn[top, j + 1] = _ms(
                            R[top, j],
                            L[bot, j + 1] + R[bot, j],
                            self.alpha,
                        )
                        Rn[bot, j + 1] = R[bot, j] + _ms(
                            L[top, j + 1], R[top, j], self.alpha
                        )

            L = Ln
            R = Rn
            num_iters = it
            u_hat = self._decide(L, R)

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                break

        u_hat = self._decide(L, R)
        return u_hat, num_iters
