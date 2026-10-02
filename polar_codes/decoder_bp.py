"""
极化码 BP（置信传播）译码器
"""
import math
import numpy as np

from encoder import polar_encode, get_inverse_generator_matrix
from decoder_sc import f_operation


def _minsum_f(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（因子图 min-sum）；若不收敛则回退到 G^{-1} 硬判决"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.LARGE = 1e6
        self._Ginv = get_inverse_generator_matrix(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        idx2 = idx + s
                        L[idx, j - 1] = _minsum_f(
                            R[idx, j - 1] + L[idx2, j], L[idx, j], self.alpha
                        )
                        L[idx2, j - 1] = (
                            _minsum_f(R[idx, j - 1], L[idx, j], self.alpha) + L[idx2, j]
                        )

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        idx2 = idx + s
                        R[idx, j + 1] = _minsum_f(
                            R[idx2, j] + L[idx2, j + 1], R[idx, j], self.alpha
                        )
                        R[idx2, j + 1] = (
                            _minsum_f(R[idx, j], L[idx, j + 1], self.alpha) + R[idx2, j]
                        )

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                break

        total = L[:, 0] + R[:, 0]
        u_bp = (total < 0).astype(int)
        u_bp[self.frozen_idx] = 0

        x_hard = (llr_ch < 0).astype(int)
        u_fb = (self._Ginv @ x_hard) % 2
        u_fb[self.frozen_idx] = 0

        if np.array_equal(polar_encode(u_bp), x_hard):
            return u_bp, num_iters
        return u_fb.astype(int), num_iters
