"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode, bit_reversal_permutation


def _f_minsum(a, b, alpha):
    """Min-sum f operation with normalization."""
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


def _boxplus(a, b):
    """Exact LLR-domain box-plus (f function)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = np.clip(a, -50.0, 50.0)
    b = np.clip(b, -50.0, 50.0)
    ta, tb = np.exp(-a), np.exp(-b)
    num = 1.0 + ta * tb
    den = ta + tb
    den = np.maximum(den, 1e-300)
    return np.log(num / den)


class BPDecoder:
    """
    BP 译码器。
    """

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        """
        主译码函数。
        """
        llr_tx = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = llr_tx[self.br]

        N = self.N
        n = self.n
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(self.max_iter):
            num_iters = it + 1

            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        a = i + k
                        b = a + step
                        L[a, j - 1] = _boxplus(R[a, j] + L[b, j], L[a, j])
                        L[b, j - 1] = _f_minsum(R[a, j], L[a, j], alpha) + L[b, j]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        a = i + k
                        b = a + step
                        R[a, j + 1] = _boxplus(R[b, j] + L[b, j + 1], R[a, j])
                        R[b, j + 1] = _f_minsum(R[a, j], L[a, j + 1], alpha) + R[b, j]

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            x_hard = (llr_tx < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
