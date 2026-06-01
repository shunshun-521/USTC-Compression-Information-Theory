"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停
"""
import numpy as np
from encoder import bit_reversal_permutation, polar_encode
from decoder_sc import f_operation


def _minsum_f(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（因子图列 0..n）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.rev = bit_reversal_permutation(N)
        self.large = 1e6

    def decode(self, llr_ch):
        """BP 译码，返回 (u_hat, num_iters)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = llr_ch[self.rev].copy()
        n, N = self.n, self.N
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        hard_ch = (llr_ch < 0).astype(int)
        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        idx = i + t
                        La = R[idx, j] + L[idx, j + 1]
                        Lb = R[idx + s, j] + L[idx + s, j + 1]
                        L[idx, j] = _minsum_f(La, Lb, alpha)
                        L[idx + s, j] = _minsum_f(
                            R[idx, j], L[idx, j + 1], alpha
                        ) + L[idx + s, j + 1]

            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        idx = i + t
                        R[idx, j] = _minsum_f(
                            R[idx + s, j - 1] + L[idx + s, j],
                            R[idx, j - 1],
                            alpha,
                        )
                        R[idx + s, j] = (
                            _minsum_f(R[idx, j - 1], L[idx, j], alpha)
                            + R[idx + s, j - 1]
                        )

            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[total < 0] = 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
        else:
            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[total < 0] = 1
            u_hat[self.frozen_bits] = 0

        return u_hat, num_iters
