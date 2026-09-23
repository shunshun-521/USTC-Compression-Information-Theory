"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math

from encoder import bit_reversal_permutation, polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self.LARGE = 1e6

    def decode(self, llr_ch):
        """
        主译码函数。
        返回: (u_hat, num_iters)
        """
        N = self.N
        n = self.n
        alpha = self.alpha
        llr = llr_ch[self.br].astype(np.float64)

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_bits == 1, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for stage in range(n - 1, -1, -1):
                step = 1 << stage
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx = i + k
                        col = stage + 1
                        L[idx, stage] = _f_min_sum(
                            R[idx, col] + L[idx + step, col],
                            L[idx, col], alpha
                        )
                        L[idx + step, stage] = _f_min_sum(
                            R[idx, col], L[idx, col], alpha
                        ) + L[idx + step, col]

            for stage in range(n):
                step = 1 << stage
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx = i + k
                        col = stage + 1
                        R[idx, col] = _f_min_sum(
                            R[idx + step, stage] + L[idx + step, col],
                            R[idx, stage], alpha
                        )
                        R[idx + step, col] = _f_min_sum(
                            R[idx, stage], L[idx, col], alpha
                        ) + R[idx + step, stage]

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                num_iters = it
                break

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
