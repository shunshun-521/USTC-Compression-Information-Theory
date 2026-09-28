"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（分层 L/R 消息传递）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_nat = np.asarray(llr_ch, dtype=np.float64)
        llr_work = llr_nat[self.br]
        n = self.n
        N = self.N
        damping = 0.75

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_work
        R[0, :] = 0.0
        R[0, self.frozen_bits] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for layer in range(n - 1, -1, -1):
                step = 1 << layer
                for i in range(0, N, 2 * step):
                    la = L[layer + 1, i : i + step]
                    lb = L[layer + 1, i + step : i + 2 * step]
                    ra = R[layer, i : i + step]
                    new_la = _f_min_sum(ra + lb, la, self.alpha)
                    new_lb = _f_min_sum(ra, la, self.alpha) + lb
                    if it > 1:
                        L[layer, i : i + step] = damping * L[layer, i : i + step] + (1 - damping) * new_la
                        L[layer, i + step : i + 2 * step] = (
                            damping * L[layer, i + step : i + 2 * step] + (1 - damping) * new_lb
                        )
                    else:
                        L[layer, i : i + step] = new_la
                        L[layer, i + step : i + 2 * step] = new_lb

            for layer in range(0, n):
                step = 1 << layer
                for i in range(0, N, 2 * step):
                    ra = R[layer, i : i + step]
                    rb = R[layer, i + step : i + 2 * step]
                    la = L[layer + 1, i : i + step]
                    lb = L[layer + 1, i + step : i + 2 * step]
                    new_ra = _f_min_sum(rb + lb, ra, self.alpha)
                    new_rb = _f_min_sum(ra, la, self.alpha) + rb
                    if it > 1:
                        R[layer + 1, i : i + step] = (
                            damping * R[layer + 1, i : i + step] + (1 - damping) * new_ra
                        )
                        R[layer + 1, i + step : i + 2 * step] = (
                            damping * R[layer + 1, i + step : i + 2 * step] + (1 - damping) * new_rb
                        )
                    else:
                        R[layer + 1, i : i + step] = new_ra
                        R[layer + 1, i + step : i + 2 * step] = new_rb

            total = L[0, :] + R[0, :]
            for i in range(N):
                u_hat[i] = 0 if total[i] >= 0 else 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_nat < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[0, :] + R[0, :]
        for i in range(N):
            u_hat[i] = 0 if total[i] >= 0 else 1
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
