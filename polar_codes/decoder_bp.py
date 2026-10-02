"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, _bitrev_indices


def _minsum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        if self.frozen_bits.dtype != bool:
            self.frozen_bits = self.frozen_bits.astype(int) > 0
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6
        self.br = _bitrev_indices(N)
        self.inv_br = np.argsort(self.br)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_dec = llr_ch[self.inv_br]

        L = np.zeros((self.N, self.n + 1), dtype=np.float64)
        R = np.zeros((self.N, self.n + 1), dtype=np.float64)
        L[:, self.n] = llr_dec
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(self.N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            for j in range(self.n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, self.N, 2 * s):
                    L[i : i + s, j - 1] = _minsum(
                        R[i : i + s, j] + L[i + s : i + 2 * s, j],
                        L[i : i + s, j],
                        self.alpha,
                    )
                    L[i + s : i + 2 * s, j - 1] = (
                        _minsum(R[i : i + s, j], L[i : i + s, j], self.alpha)
                        + L[i + s : i + 2 * s, j]
                    )

            for j in range(0, self.n):
                s = 1 << j
                for i in range(0, self.N, 2 * s):
                    R[i : i + s, j + 1] = _minsum(
                        R[i + s : i + 2 * s, j] + L[i + s : i + 2 * s, j + 1],
                        R[i : i + s, j],
                        self.alpha,
                    )
                    R[i + s : i + 2 * s, j + 1] = (
                        _minsum(R[i : i + s, j], L[i : i + s, j + 1], self.alpha)
                        + R[i + s : i + 2 * s, j]
                    )

            for i in range(self.N):
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(self.N):
            u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
        u_hat[self.frozen_bits] = 0
        return u_hat.astype(int), num_iters
