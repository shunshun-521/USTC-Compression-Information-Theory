"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation

LARGE = 1e6


def _minsum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        alpha = self.alpha

        # L[i,j], R[i,j]: shape (N, n+1), j=0..n
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        llr_br = llr_ch[self.br]
        L[:, n] = llr_br
        R[:, 0] = 0.0
        frozen_br = self.frozen_bits[self.br]
        R[frozen_br, 0] = LARGE

        num_iters = 0
        for it in range(self.max_iter):
            num_iters = it + 1
            # Right to left: update L
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    L[i, j - 1] = _minsum(
                        R[i, j] + L[i + s, j], L[i, j + 1], alpha
                    )
                    L[i + s, j - 1] = _minsum(R[i, j], L[i, j + 1], alpha) + L[
                        i + s, j + 1
                    ]

            # Left to right: update R
            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    R[i, j] = _minsum(
                        R[i + s, j] + L[i + s, j], R[i, j - 1], alpha
                    )
                    R[i + s, j] = _minsum(R[i, j - 1], L[i, j], alpha) + R[i + s, j - 1]

            # Early stopping
            u_hat_br = np.zeros(N, dtype=int)
            for i in range(N):
                if frozen_br[i]:
                    u_hat_br[i] = 0
                else:
                    total = L[i, 0] + R[i, 0]
                    u_hat_br[i] = 0 if total >= 0 else 1

            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                u_hat[self.br[i]] = u_hat_br[i]

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                break

        u_hat_br = np.zeros(N, dtype=int)
        for i in range(N):
            if frozen_br[i]:
                u_hat_br[i] = 0
            else:
                total = L[i, 0] + R[i, 0]
                u_hat_br[i] = 0 if total >= 0 else 1

        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            u_hat[self.br[i]] = u_hat_br[i]

        return u_hat, num_iters
