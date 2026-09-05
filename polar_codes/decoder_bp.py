"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from channel import hard_decision_llr
from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for stage in range(n - 1, -1, -1):
                step = 1 << stage
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx = i + k
                        idx_s = idx + step
                        L[idx, stage] = _f_min_sum(
                            R[idx, stage + 1] + L[idx_s, stage + 1],
                            L[idx, stage + 1],
                            self.alpha,
                        )
                        L[idx_s, stage] = (
                            _f_min_sum(
                                R[idx, stage + 1], L[idx, stage + 1], self.alpha
                            )
                            + L[idx_s, stage + 1]
                        )

            for stage in range(1, n):
                step = 1 << (stage - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx = i + k
                        idx_s = idx + step
                        R[idx, stage] = _f_min_sum(
                            R[idx_s, stage] + L[idx_s, stage + 1],
                            R[idx, stage - 1],
                            self.alpha,
                        )
                        R[idx_s, stage] = (
                            _f_min_sum(R[idx, stage - 1], L[idx, stage + 1], self.alpha)
                            + R[idx_s, stage]
                        )

            if n > 0:
                step = 1 << (n - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx = i + k
                        idx_s = idx + step
                        R[idx, n] = _f_min_sum(
                            R[idx_s, n] + L[idx_s, n], R[idx, n - 1], self.alpha
                        )
                        R[idx_s, n] = (
                            _f_min_sum(R[idx, n - 1], L[idx, n], self.alpha) + R[idx_s, n]
                        )

            total_llr = L[:, 0] + R[:, 0]
            u_hat = (total_llr < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                break

        total_llr = L[:, 0] + R[:, 0]
        u_hat = (total_llr < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
