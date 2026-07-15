"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import _reorder_channel_llr
from encoder import polar_encode


def _bp_f(x, y, alpha):
    """min-sum f 运算"""
    s1 = 1 if np.sign(x) == 0 else np.sign(x)
    s2 = 1 if np.sign(y) == 0 else np.sign(y)
    return alpha * s1 * s2 * min(abs(x), abs(y))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        if self.frozen_bits.dtype == bool:
            self.frozen_mask = self.frozen_bits
        else:
            self.frozen_mask = self.frozen_bits.astype(bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, num_iters"""
        llr_ch = _reorder_channel_llr(llr_ch)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_mask, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, s << 1):
                    for k in range(s):
                        idx = i + k
                        idx2 = idx + s
                        L[idx, j - 1] = _bp_f(
                            R[idx, j] + L[idx2, j],
                            L[idx, j],
                            self.alpha,
                        )
                        L[idx2, j - 1] = _bp_f(
                            R[idx, j],
                            L[idx, j],
                            self.alpha,
                        ) + L[idx2, j]

            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for i in range(0, N, s << 1):
                    for k in range(s):
                        idx = i + k
                        idx2 = idx + s
                        R[idx, j] = _bp_f(
                            R[idx2, j] + L[idx2, j],
                            R[idx, j - 1],
                            self.alpha,
                        )
                        R[idx2, j] = _bp_f(
                            R[idx, j - 1],
                            L[idx, j],
                            self.alpha,
                        ) + R[idx2, j]

            num_iters = it

            for i in range(N):
                if self.frozen_mask[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            if self.frozen_mask[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
