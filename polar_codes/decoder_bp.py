"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math

from encoder import polar_encode, bit_reversal_permutation


def ms_f(x, y, alpha=0.9375):
    """Min-sum f 运算（带缩放因子 alpha）。"""
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        if 2 ** self.n != N:
            raise ValueError(f"N={N} must be power of 2")
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.rev = bit_reversal_permutation(N)
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.LARGE = 1e6

    def decode(self, llr_ch):
        """主译码函数。返回：u_hat, num_iters"""
        n = self.n
        N = self.N
        llr = np.asarray(llr_ch, dtype=np.float64)[self.rev]

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for lam in range(n - 1, -1, -1):
                span = 1 << lam
                for i in range(0, N, 2 * span):
                    for k in range(span):
                        idx = i + k
                        L[idx, lam] = ms_f(
                            R[idx, lam + 1] + L[idx + span, lam + 1],
                            L[idx, lam + 1],
                            self.alpha,
                        )
                        L[idx + span, lam] = ms_f(
                            R[idx, lam + 1],
                            L[idx, lam + 1],
                            self.alpha,
                        ) + L[idx + span, lam + 1]

            for lam in range(0, n):
                span = 1 << lam
                for i in range(0, N, 2 * span):
                    for k in range(span):
                        idx = i + k
                        R[idx, lam + 1] = ms_f(
                            R[idx + span, lam] + L[idx + span, lam + 1],
                            R[idx, lam],
                            self.alpha,
                        )
                        R[idx + span, lam + 1] = ms_f(
                            R[idx, lam],
                            L[idx, lam + 1],
                            self.alpha,
                        ) + R[idx + span, lam]

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                num_iters = it
                break
            num_iters = it

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat.astype(int), num_iters
