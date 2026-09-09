"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math

from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    """Min-sum f operation with scaling factor alpha."""
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    """

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        # L[i][j]: left message at row i, column j (0..n)
        # R[i][j]: right message at row i, column j (0..n)
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # Right-to-left: update L messages (columns n down to 1)
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_top = i + k
                        idx_btm = i + k + s
                        L[idx_top, j - 1] = _f_min_sum(
                            R[idx_top, j] + L[idx_btm, j],
                            L[idx_top, j],
                            self.alpha,
                        )
                        L[idx_btm, j - 1] = (
                            _f_min_sum(R[idx_top, j], L[idx_top, j], self.alpha)
                            + L[idx_btm, j]
                        )

            # Left-to-right: update R messages (columns 0 to n-1)
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_top = i + k
                        idx_btm = i + k + s
                        R[idx_top, j + 1] = _f_min_sum(
                            R[idx_btm, j] + L[idx_btm, j + 1],
                            R[idx_top, j],
                            self.alpha,
                        )
                        R[idx_btm, j + 1] = (
                            _f_min_sum(R[idx_top, j], L[idx_top, j + 1], self.alpha)
                            + R[idx_btm, j]
                        )

            num_iters = it

            # Early stopping
            total_llr = L[:, 0] + R[:, 0]
            u_hat = (total_llr < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                break

        # Final decision
        total_llr = L[:, 0] + R[:, 0]
        u_hat = (total_llr < 0).astype(int)
        u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
