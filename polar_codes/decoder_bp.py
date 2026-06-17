"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode


def _minsum_f(a, b, alpha):
    return alpha * f_operation(a, b)


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 u_hat, num_iters
        """
        N, n = self.N, self.n
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_natural = llr_ch.copy()
        from decoder_sc import _remap_channel_llrs
        llr_ch = _remap_channel_llrs(llr_ch)

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        u_hat = np.zeros(N, dtype=int)
        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        i0 = i + t
                        i1 = i + t + s
                        L[i0, j - 1] = _minsum_f(
                            R[i0, j] + L[i1, j],
                            L[i0, j],
                            self.alpha,
                        )
                        L[i1, j - 1] = _minsum_f(R[i0, j], L[i0, j], self.alpha) + L[i1, j]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        i0 = i + t
                        i1 = i + t + s
                        R[i0, j + 1] = _minsum_f(
                            R[i1, j] + L[i1, j + 1],
                            R[i0, j],
                            self.alpha,
                        )
                        R[i1, j + 1] = _minsum_f(R[i0, j], L[i0, j + 1], self.alpha) + R[i1, j]

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_natural < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
