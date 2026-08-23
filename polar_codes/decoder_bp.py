"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode


def _f_min_sum(x, y, alpha):
    if x == 0.0 or y == 0.0:
        return 0.0
    return alpha * np.sign(x) * np.sign(y) * min(abs(x), abs(y))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.large

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            for stage in range(n, 0, -1):
                span = 1 << (stage - 1)
                for block in range(0, N, span << 1):
                    for j in range(span):
                        i = block + j
                        ip = i + span
                        L[i, stage - 1] = _f_min_sum(
                            R[i, stage] + L[ip, stage],
                            L[i, stage],
                            self.alpha,
                        )
                        L[ip, stage - 1] = _f_min_sum(R[i, stage], L[i, stage], self.alpha) + L[ip, stage]

            for stage in range(1, n + 1):
                span = 1 << (stage - 1)
                for block in range(0, N, span << 1):
                    for j in range(span):
                        i = block + j
                        ip = i + span
                        R[i, stage - 1] = _f_min_sum(
                            R[ip, stage] + L[ip, stage],
                            R[i, stage - 1],
                            self.alpha,
                        )
                        R[ip, stage - 1] = _f_min_sum(R[i, stage - 1], L[i, stage], self.alpha) + R[ip, stage]

            num_iters = it
            u_hat = self._hard_decision(L, R)
            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters

    def _hard_decision(self, L, R):
        u_hat = (L[:, 0] + R[:, 0] < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat
