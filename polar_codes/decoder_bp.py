"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _f_minsum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（因子图列 0..n，信道 LLR 在列 n）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = int(max_iter)
        self.alpha = float(alpha)
        self.br = bit_reversal_permutation(N)
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_perm = llr_ch[self.br]
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_perm
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            # 右到左：列 n-1 .. 0
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        idx = i + t
                        L[idx, j] = _f_minsum(
                            R[idx, j + 1] + L[idx + s, j + 1],
                            L[idx, j + 1],
                            self.alpha,
                        )
                        L[idx + s, j] = (
                            _f_minsum(R[idx, j + 1], L[idx, j + 1], self.alpha)
                            + L[idx + s, j + 1]
                        )

            # 左到右：列 1 .. n
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        idx = i + t
                        R[idx, j + 1] = _f_minsum(
                            R[idx + s, j] + L[idx + s, j + 1],
                            R[idx, j],
                            self.alpha,
                        )
                        R[idx + s, j + 1] = (
                            _f_minsum(R[idx, j], L[idx, j + 1], self.alpha)
                            + R[idx + s, j]
                        )

            num_iters = it
            total = L[:, 0] + R[:, 0]
            u_stage = np.where(total >= 0, 0, 1).astype(np.int8)
            u_stage[self.frozen_bits] = 0

            x_hat = polar_encode(u_stage)
            hard_ch = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard_ch):
                u_hat = u_stage
                break
            u_hat = u_stage

        return u_hat, num_iters
