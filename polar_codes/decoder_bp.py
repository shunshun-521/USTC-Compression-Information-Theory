"""
极化码 BP（置信传播）译码器
min-sum 近似 + 早停
"""
import numpy as np
from encoder import bit_reversal_permutation, polar_encode


def _ms(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器（因子图列 0..n）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.max_iter = max_iter
        self.alpha = alpha
        frozen_bits = np.asarray(frozen_bits)
        self.frozen = np.zeros(N, dtype=bool)
        if frozen_bits.dtype == bool:
            self.frozen = frozen_bits.copy()
        else:
            self.frozen = frozen_bits.astype(int) > 0
        self.br = bit_reversal_permutation(N)
        self.large = 1e6

    def decode(self, llr_ch):
        llr_in = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = llr_in[self.br]
        n, N = self.n, self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen, 0] = self.large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, step * 2):
                    L[i, j - 1] = _ms(
                        R[i, j] + L[i + step, j], L[i, j], self.alpha
                    )
                    L[i + step, j - 1] = _ms(R[i, j], L[i, j], self.alpha) + L[
                        i + step, j
                    ]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, step * 2):
                    R[i, j + 1] = _ms(
                        R[i + step, j] + L[i + step, j + 1], R[i, j], self.alpha
                    )
                    R[i + step, j + 1] = (
                        _ms(R[i, j], L[i, j + 1], self.alpha) + R[i + step, j]
                    )

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen] = 0
            x_hat = polar_encode(u_hat)
            hard_x = (llr_in < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break
        else:
            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen] = 0

        return u_hat, num_iters
