"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制

因子图列索引：0 = 信道端，n = 信源端
"""
import math

import numpy as np

from encoder import polar_encode


def bp_f(x, y, alpha=0.9375):
    """min-sum 近似的 f 运算。"""
    return alpha * np.sign(x) * np.sign(y) * min(abs(x), abs(y))


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e7

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def _hard_decision(self, Ln, Rn):
        total = Ln + Rn
        u_hat = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total[i] >= 0 else 1
        return u_hat

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        alpha = self.alpha

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)

        L[0, :] = llr_ch
        R[n, :] = 0.0
        R[n, self.frozen_bits] = self.LARGE

        num_iters = 0
        for it in range(self.max_iter):
            num_iters = it + 1

            for stage in range(0, n):
                stride = 1 << stage
                block_size = 1 << (stage + 1)
                for base in range(0, N, block_size):
                    for offset in range(stride):
                        i0 = base + offset
                        i1 = i0 + stride
                        if stage == 0:
                            L[stage + 1, i0] = bp_f(
                                R[stage + 1, i1] + L[stage, i1],
                                L[stage, i0],
                                alpha,
                            )
                        else:
                            L[stage + 1, i0] = bp_f(
                                L[stage, i1],
                                L[stage, i0],
                                alpha,
                            )
                        L[stage + 1, i1] = bp_f(
                            R[stage + 1, i0],
                            L[stage, i0],
                            alpha,
                        ) + L[stage, i1]

            for stage in range(n - 1, -1, -1):
                stride = 1 << stage
                block_size = 1 << (stage + 1)
                for base in range(0, N, block_size):
                    for offset in range(stride):
                        i0 = base + offset
                        i1 = i0 + stride
                        R[stage, i0] = bp_f(
                            R[stage + 1, i0],
                            R[stage + 1, i1] + L[stage, i1],
                            alpha,
                        )
                        R[stage, i1] = R[stage + 1, i1] + bp_f(
                            R[stage + 1, i0],
                            L[stage, i0],
                            alpha,
                        )

            u_hat = self._hard_decision(L[n, :], R[n, :])
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        u_hat = self._hard_decision(L[n, :], R[n, :])
        return u_hat, num_iters
