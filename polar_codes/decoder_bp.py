"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode


def _minsum_f(a, b, alpha):
    """min-sum box-plus 近似"""
    return alpha * f_operation(a, b)


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        alpha = self.alpha

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)

        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_idx] = self.LARGE

        num_iters = self.max_iter

        for iteration in range(1, self.max_iter + 1):
            for s in range(n):
                half = 1 << s
                block = half << 1
                for i in range(0, N, block):
                    for k in range(half):
                        idx = i + k
                        idx2 = idx + half
                        R[s + 1, idx] = _minsum_f(
                            R[s, idx],
                            L[s + 1, idx2] + R[s, idx2],
                            alpha,
                        )
                        R[s + 1, idx2] = (
                            _minsum_f(R[s, idx], L[s + 1, idx], alpha)
                            + R[s, idx2]
                        )

            for s in range(n - 1, -1, -1):
                half = 1 << s
                block = half << 1
                for i in range(0, N, block):
                    for k in range(half):
                        idx = i + k
                        idx2 = idx + half
                        L[s, idx] = _minsum_f(
                            L[s + 1, idx],
                            L[s + 1, idx2] + R[s, idx2],
                            alpha,
                        )
                        L[s, idx2] = (
                            _minsum_f(R[s, idx], L[s + 1, idx], alpha)
                            + L[s + 1, idx2]
                        )

            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                total = L[0, i] + R[0, i]
                u_hat[i] = 0 if total >= 0 else 1
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = iteration
                break

        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            total = L[0, i] + R[0, i]
            u_hat[i] = 0 if total >= 0 else 1
        u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
