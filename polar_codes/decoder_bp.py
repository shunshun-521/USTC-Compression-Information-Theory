"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode


def _boxplus_minsum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * min(abs(a), abs(b))


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_indices = np.where(self.frozen_bits == 1)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        br = np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)
        llr_ch = llr_ch[br]

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_indices, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx = i + k
                        L[idx, j - 1] = _boxplus_minsum(
                            R[idx, j - 1] + L[idx + step, j],
                            L[idx, j],
                            self.alpha,
                        )
                        L[idx + step, j - 1] = _boxplus_minsum(
                            R[idx, j - 1],
                            L[idx, j],
                            self.alpha,
                        ) + L[idx + step, j]

            for j in range(1, n + 1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx = i + k
                        R[idx, j - 1] = _boxplus_minsum(
                            R[idx + step, j] + L[idx + step, j],
                            R[idx, j - 1],
                            self.alpha,
                        )
                        R[idx + step, j - 1] = _boxplus_minsum(
                            R[idx, j - 1],
                            L[idx, j],
                            self.alpha,
                        ) + R[idx + step, j]

            u_hat = np.where((L[:, 0] + R[:, 0]) >= 0, 0, 1).astype(int)
            u_hat[self.frozen_indices] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        u_hat = np.where((L[:, 0] + R[:, 0]) >= 0, 0, 1).astype(int)
        u_hat[self.frozen_indices] = 0
        return u_hat, num_iters
