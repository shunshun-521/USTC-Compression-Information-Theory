"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode, bit_reversal_permutation


def _ms_f(x, y, alpha):
    """min-sum f 运算。"""
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_indices = set(np.where(self.frozen_bits)[0])
        self.max_iter = max_iter
        self.alpha = alpha
        self.brp = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        n = self.n
        N = self.N
        llr = np.asarray(llr_ch, dtype=np.float64)[self.brp]

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr
        R[:, 0] = 0.0
        for idx in self.frozen_indices:
            R[idx, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for stage in range(n - 1, -1, -1):
                span = 2 ** stage
                for base in range(0, N, 2 * span):
                    for k in range(span):
                        i = base + k
                        j = base + k + span
                        L[i, stage] = _ms_f(
                            R[i, stage + 1] + L[j, stage + 1],
                            L[i, stage + 1],
                            self.alpha,
                        )
                        L[j, stage] = _ms_f(
                            R[i, stage + 1],
                            L[i, stage + 1],
                            self.alpha,
                        ) + L[j, stage + 1]

            for stage in range(n):
                span = 2 ** stage
                for base in range(0, N, 2 * span):
                    for k in range(span):
                        i = base + k
                        j = base + k + span
                        R[i, stage + 1] = _ms_f(
                            R[j, stage + 1] + L[j, stage + 1],
                            R[i, stage],
                            self.alpha,
                        )
                        R[j, stage + 1] = _ms_f(
                            R[i, stage],
                            L[j, stage + 1],
                            self.alpha,
                        ) + R[j, stage + 1]

            for i in range(N):
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
            for idx in self.frozen_indices:
                u_hat[idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (np.asarray(llr_ch, dtype=np.float64) < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        for i in range(N):
            u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
        for idx in self.frozen_indices:
            u_hat[idx] = 0

        return u_hat, num_iters
