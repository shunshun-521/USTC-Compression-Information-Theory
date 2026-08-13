"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode


def _minsum_f(a, b, alpha=0.9375):
    """min-sum 近似的 f 运算。"""
    sa = 1.0 if a >= 0 else -1.0
    sb = 1.0 if b >= 0 else -1.0
    return alpha * sa * sb * min(abs(a), abs(b))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.LARGE = 1e6

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        idx2 = idx + s
                        L[idx, j] = _minsum_f(
                            R[idx, j + 1] + L[idx2, j + 1],
                            L[idx, j + 1],
                            self.alpha,
                        )
                        L[idx2, j] = _minsum_f(
                            R[idx, j + 1], L[idx, j + 1], self.alpha
                        ) + L[idx2, j + 1]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        idx2 = idx + s
                        R[idx, j + 1] = _minsum_f(
                            R[idx2, j] + L[idx2, j + 1], R[idx, j], self.alpha
                        )
                        R[idx2, j + 1] = _minsum_f(
                            R[idx, j], L[idx, j + 1], self.alpha
                        ) + R[idx2, j]

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            u_hat[i] = 0 if total >= 0 else 1
        u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
