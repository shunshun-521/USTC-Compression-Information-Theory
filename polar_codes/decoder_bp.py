"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode


def _boxplus_minsum(a, b, alpha=0.9375):
    """min-sum 近似的 f 运算"""
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]
        self._large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self._large

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int32)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, step << 1):
                    s = step
                    La = R[i:i + s, j - 1] + L[i + s:i + 2 * s, j]
                    Lb = L[i:i + s, j]
                    L[i:i + s, j - 1] = _boxplus_minsum(La, Lb, self.alpha)
                    L[i + s:i + 2 * s, j - 1] = (
                        _boxplus_minsum(R[i:i + s, j - 1], Lb, self.alpha) + L[i + s:i + 2 * s, j]
                    )

            for j in range(1, n + 1):
                step = 1 << (j - 1)
                for i in range(0, N, step << 1):
                    s = step
                    Ra = R[i + s:i + 2 * s, j] + L[i + s:i + 2 * s, j]
                    Rb = R[i:i + s, j - 1]
                    Lb = L[i:i + s, j]
                    R[i:i + s, j - 1] = _boxplus_minsum(Ra, Lb, self.alpha)
                    R[i + s:i + 2 * s, j - 1] = (
                        _boxplus_minsum(Rb, Lb, self.alpha) + R[i + s:i + 2 * s, j]
                    )

            num_iters = it
            total_llr = L[:, 0] + R[:, 0]
            u_hat = (total_llr < 0).astype(np.int32)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(np.int32)
            if np.array_equal(x_hat, hard_ch):
                break

        total_llr = L[:, 0] + R[:, 0]
        u_hat = (total_llr < 0).astype(np.int32)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
