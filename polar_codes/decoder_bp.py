"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from channel import hard_decision_llr
from decoder_sc import f_operation
from encoder import bit_reversal_permutation, polar_encode


def _minsum_f(x, y, alpha):
    """min-sum f with scaling factor alpha."""
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """
    BP 译码器。
    """

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        N = self.N
        n = self.n
        alpha = self.alpha

        llr_perm = llr_ch[self.br].astype(np.float64)

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_perm
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, step * 2):
                    s = step
                    for k in range(step):
                        idx = i + k
                        idx2 = i + k + s
                        L[idx, j - 1] = _minsum_f(
                            R[idx, j] + L[idx2, j],
                            L[idx, j],
                            alpha,
                        )
                        L[idx2, j - 1] = _minsum_f(
                            R[idx, j],
                            L[idx, j],
                            alpha,
                        ) + L[idx2, j]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, step * 2):
                    s = step
                    for k in range(step):
                        idx = i + k
                        idx2 = i + k + s
                        R[idx, j + 1] = _minsum_f(
                            R[idx2, j] + L[idx2, j + 1],
                            R[idx, j],
                            alpha,
                        )
                        R[idx2, j + 1] = _minsum_f(
                            R[idx, j],
                            L[idx, j + 1],
                            alpha,
                        ) + R[idx2, j]

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    total = L[i, 0] + R[i, 0]
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_decision_llr(llr_ch)):
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
