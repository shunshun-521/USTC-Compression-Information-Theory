"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode


def _f_min_sum(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器，因子图 n+1 列，每列 N 个节点。"""

    LARGE = 1e9

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for stage in range(n - 1, -1, -1):
                sp = 1 << stage
                for phi in range(0, N, 2 * sp):
                    for beta in range(sp):
                        li = phi + beta
                        ri = phi + beta + sp
                        L[li, stage] = _f_min_sum(
                            R[li, stage] + L[ri, stage + 1],
                            L[li, stage + 1],
                            self.alpha,
                        )
                        L[ri, stage] = _f_min_sum(
                            R[li, stage],
                            L[li, stage + 1],
                            self.alpha,
                        ) + L[ri, stage + 1]

            for stage in range(n):
                sp = 1 << stage
                for phi in range(0, N, 2 * sp):
                    for beta in range(sp):
                        li = phi + beta
                        ri = phi + beta + sp
                        R[li, stage + 1] = _f_min_sum(
                            R[ri, stage] + L[ri, stage + 1],
                            R[li, stage],
                            self.alpha,
                        )
                        R[ri, stage + 1] = _f_min_sum(
                            R[li, stage],
                            L[li, stage + 1],
                            self.alpha,
                        ) + R[ri, stage]

            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[total < 0] = 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[total < 0] = 1
        u_hat[self.frozen_bits] = 0

        return u_hat, num_iters
