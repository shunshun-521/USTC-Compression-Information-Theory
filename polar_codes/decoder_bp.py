"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math

from encoder import polar_encode


def _boxplus_min_sum(a, b, alpha):
    """min-sum 近似的 boxplus。"""
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n, alpha = self.N, self.n, self.alpha

        # stage 0 = 信源端，stage n = 信道端
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(self.max_iter):
            num_iters = it + 1

            # 从信道向信源传播 L（stage n-1 到 0）
            for stage in range(n - 1, -1, -1):
                step = 1 << stage
                for r in range(0, N, 2 * step):
                    for j in range(step):
                        i = r + j
                        L[i, stage] = _boxplus_min_sum(
                            L[i, stage + 1],
                            L[i + step, stage + 1] + R[i + step, stage],
                            alpha,
                        )
                        L[i + step, stage] = _boxplus_min_sum(
                            L[i, stage + 1],
                            R[i, stage],
                            alpha,
                        ) + L[i + step, stage + 1]

            # 从信源向信道传播 R（stage 0 到 n-1）
            for stage in range(n):
                step = 1 << stage
                for r in range(0, N, 2 * step):
                    for j in range(step):
                        i = r + j
                        R[i, stage + 1] = _boxplus_min_sum(
                            R[i, stage],
                            R[i + step, stage] + L[i + step, stage + 1],
                            alpha,
                        )
                        R[i + step, stage + 1] = _boxplus_min_sum(
                            R[i, stage],
                            L[i, stage + 1],
                            alpha,
                        ) + R[i + step, stage]

            total = L[:, 0] + R[:, 0]
            u_hat = np.where(total >= 0, 0, 1).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.where(total >= 0, 0, 1).astype(int)
        u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
