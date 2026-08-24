"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from channel import hard_decision_llr


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（极化码蝶形因子图）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.LARGE = 1e6

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)

        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_idx] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for layer in range(n, 0, -1):
                step = 1 << (layer - 1)
                for block in range(0, N, 2 * step):
                    for j in range(step):
                        li = block + j
                        ri = block + j + step
                        L[layer - 1, li] = _f_min_sum(
                            R[layer, li] + L[layer, ri],
                            L[layer, li],
                            self.alpha,
                        )
                        L[layer - 1, ri] = (
                            _f_min_sum(R[layer, li], L[layer, li], self.alpha)
                            + L[layer, ri]
                        )

            for layer in range(0, n):
                step = 1 << layer
                for block in range(0, N, 2 * step):
                    for j in range(step):
                        li = block + j
                        ri = block + j + step
                        R[layer + 1, li] = _f_min_sum(
                            R[layer, ri] + L[layer + 1, ri],
                            R[layer, li],
                            self.alpha,
                        )
                        R[layer + 1, ri] = (
                            _f_min_sum(R[layer, li], L[layer + 1, li], self.alpha)
                            + R[layer, ri]
                        )

            num_iters = it
            for i in range(N):
                u_hat[i] = 0 if self.frozen_bits[i] else (
                    0 if (L[0, i] + R[0, i]) >= 0 else 1
                )

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                break

        for i in range(N):
            u_hat[i] = 0 if self.frozen_bits[i] else (
                0 if (L[0, i] + R[0, i]) >= 0 else 1
            )

        return u_hat, num_iters
