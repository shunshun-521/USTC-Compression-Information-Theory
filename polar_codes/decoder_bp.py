"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _ms_f(a, b, alpha):
    """min-sum f 运算。"""
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """
    BP 译码器。
    """

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def _hard_decision(self, L, R):
        total = L[0, :] + R[0, :]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat

    def _check_early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, hard_ch)

    def decode(self, llr_ch):
        """
        主译码函数。
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N
        alpha = self.alpha

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_idx] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for stage in reversed(range(n)):
                step = 1 << stage
                for block in range(0, N, step << 1):
                    for i in range(step):
                        top = block + i
                        bot = top + step
                        L[stage, top] = _ms_f(
                            R[stage, top] + L[stage + 1, bot],
                            L[stage + 1, top],
                            alpha,
                        )
                        L[stage, bot] = _ms_f(
                            R[stage, top],
                            L[stage + 1, top],
                            alpha,
                        ) + L[stage + 1, bot]

            for stage in range(n):
                step = 1 << stage
                for block in range(0, N, step << 1):
                    for i in range(step):
                        top = block + i
                        bot = top + step
                        R[stage + 1, top] = _ms_f(
                            R[stage, bot] + L[stage + 1, bot],
                            R[stage, top],
                            alpha,
                        )
                        R[stage + 1, bot] = _ms_f(
                            R[stage, top],
                            L[stage + 1, top],
                            alpha,
                        ) + R[stage, bot]

            u_hat = self._hard_decision(L, R)
            if self._check_early_stop(u_hat, llr_ch):
                num_iters = it
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters
