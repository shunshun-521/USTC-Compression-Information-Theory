"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from channel import hard_decision_llr
from decoder_sc import _prepare_llr
from encoder import polar_encode


def ms_f(a, b, alpha=0.9375):
    """min-sum 近似的 f 运算。"""
    sign_a = np.where(np.sign(a) == 0, 1.0, np.sign(a))
    sign_b = np.where(np.sign(b) == 0, 1.0, np.sign(b))
    return alpha * sign_a * sign_b * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """
    BP 译码器。
    """

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]

    def decode(self, llr_ch):
        """
        主译码函数。
        """
        llr_natural = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = _prepare_llr(llr_natural)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            L[:, n] = llr_ch

            for layer in range(n - 1, -1, -1):
                step = 1 << layer
                for block in range(0, N, 2 * step):
                    for i in range(step):
                        u = block + i
                        v = block + i + step
                        L[u, layer] = ms_f(
                            R[u, layer + 1] + L[v, layer + 1],
                            L[u, layer + 1],
                            self.alpha,
                        )
                        L[v, layer] = ms_f(
                            R[u, layer + 1],
                            L[u, layer + 1],
                            self.alpha,
                        ) + L[v, layer + 1]

            for layer in range(0, n):
                step = 1 << layer
                for block in range(0, N, 2 * step):
                    for i in range(step):
                        u = block + i
                        v = block + i + step
                        R[u, layer + 1] = ms_f(
                            R[v, layer] + L[v, layer + 1],
                            R[u, layer],
                            self.alpha,
                        )
                        R[v, layer + 1] = ms_f(
                            R[u, layer],
                            L[v, layer + 1],
                            self.alpha,
                        ) + R[v, layer]

            total = L[:, 0] + R[:, 0]
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total[i] >= 0 else 1

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_natural)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                return u_hat, num_iters

        total = L[:, 0] + R[:, 0]
        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total[i] >= 0 else 1

        return u_hat, num_iters
