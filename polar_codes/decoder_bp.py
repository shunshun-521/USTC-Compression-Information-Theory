"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math

from encoder import polar_encode


def _boxplus_minsum(a, b, alpha):
    sa = np.where(a >= 0, 1.0, -1.0)
    sb = np.where(b >= 0, 1.0, -1.0)
    return alpha * sa * sb * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        if self.frozen_bits.dtype != bool:
            self.frozen_bits = self.frozen_bits.astype(bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for stage in range(n, 0, -1):
                s = 2 ** (stage - 1)
                for i in range(0, N, 2 * s):
                    for j in range(s):
                        up = i + j
                        lo = i + j + s
                        L[up, stage - 1] = _boxplus_minsum(
                            R[up, stage - 1] + L[lo, stage], L[up, stage], self.alpha
                        )
                        L[lo, stage - 1] = _boxplus_minsum(
                            R[up, stage - 1], L[up, stage], self.alpha
                        ) + L[lo, stage]

            for stage in range(1, n + 1):
                s = 2 ** (stage - 1)
                for i in range(0, N, 2 * s):
                    for j in range(s):
                        up = i + j
                        lo = i + j + s
                        R[up, stage] = _boxplus_minsum(
                            R[lo, stage] + L[lo, stage], R[up, stage - 1], self.alpha
                        )
                        R[lo, stage] = _boxplus_minsum(
                            R[up, stage - 1], L[up, stage], self.alpha
                        ) + R[lo, stage]

            total = L[:, 0] + R[:, 0]
            u_hat = np.where(total >= 0, 0, 1).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
        else:
            total = L[:, 0] + R[:, 0]
            u_hat = np.where(total >= 0, 0, 1).astype(int)
            u_hat[self.frozen_bits] = 0

        return u_hat, num_iters
