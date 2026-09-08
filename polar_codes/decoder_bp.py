"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _minsum(a, b, alpha):
    sa = np.where(a >= 0, 1.0, -1.0)
    sb = np.where(b >= 0, 1.0, -1.0)
    return alpha * sa * sb * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（分层因子图 min-sum）"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def decode(self, llr_ch):
        N = self.N
        n = self.n
        alpha = self.alpha
        llr_ch = np.clip(np.asarray(llr_ch, dtype=np.float64), -30.0, 30.0)

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for stage in range(n, 0, -1):
                step = 1 << (stage - 1)
                for i in range(0, N, 2 * step):
                    for j in range(i, i + step):
                        s = j + step
                        L[j, stage - 1] = _minsum(
                            R[j, stage] + L[s, stage], L[j, stage], alpha
                        )
                        L[s, stage - 1] = _minsum(R[j, stage], L[j, stage], alpha) + L[
                            s, stage
                        ]

            for stage in range(1, n + 1):
                step = 1 << (stage - 1)
                for i in range(0, N, 2 * step):
                    for j in range(i, i + step):
                        s = j + step
                        R[j, stage] = _minsum(
                            R[s, stage] + L[s, stage], R[j, stage - 1], alpha
                        )
                        R[s, stage] = _minsum(R[j, stage - 1], L[j, stage], alpha) + R[
                            s, stage - 1
                        ]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                num_iters = it
                break
            num_iters = it

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
