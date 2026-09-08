"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _ms_g(x, y, alpha):
    """min-sum 近似 check-node 函数。"""
    sx = np.sign(x)
    sy = np.sign(y)
    sx = 1.0 if sx == 0 else sx
    sy = 1.0 if sy == 0 else sy
    return alpha * sx * sy * min(abs(x), abs(y))


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_orig = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = llr_orig[self.br]
        n = self.n
        N = self.N
        alpha = self.alpha

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)

        L[n, :] = llr_ch
        for i in range(N):
            R[0, i] = self.LARGE if self.frozen_bits[i] else 0.0

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)
        x_hard = (llr_orig < 0).astype(int)

        for it in range(1, self.max_iter + 1):
            for s in range(n - 1, -1, -1):
                step = 1 << s
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        top = i + k
                        btm = top + step
                        L[s, top] = _ms_g(
                            L[s + 1, top],
                            L[s + 1, btm] + R[s, btm],
                            alpha,
                        )
                        L[s, btm] = (
                            _ms_g(
                                L[s + 1, top],
                                R[s, top] + L[s + 1, btm],
                                alpha,
                            )
                            + L[s + 1, btm]
                        )

            for s in range(0, n):
                step = 1 << s
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        top = i + k
                        btm = top + step
                        R[s + 1, top] = _ms_g(
                            R[s, top],
                            R[s, btm] + L[s + 1, btm],
                            alpha,
                        )
                        R[s + 1, btm] = (
                            _ms_g(
                                R[s, top],
                                L[s + 1, top] + R[s, btm],
                                alpha,
                            )
                            + R[s, btm]
                        )

            num_iters = it
            for i in range(N):
                total = L[0, i] + R[0, i]
                u_hat[i] = 0 if total >= 0 else 1
                if self.frozen_bits[i]:
                    u_hat[i] = 0

            if np.array_equal(polar_encode(u_hat), x_hard):
                break

        for i in range(N):
            total = L[0, i] + R[0, i]
            u_hat[i] = 0 if total >= 0 else 1
            if self.frozen_bits[i]:
                u_hat[i] = 0

        return u_hat, num_iters
