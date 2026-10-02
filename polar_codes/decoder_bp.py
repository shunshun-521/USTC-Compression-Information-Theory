"""
极化码 BP（置信传播）译码器
"""
import numpy as np
from encoder import polar_encode
def _f_min_sum(a, b, alpha=0.9375):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        if self.frozen.dtype != bool:
            self.frozen = self.frozen.astype(bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N
        L = np.zeros((n + 1, N))
        R = np.zeros((n + 1, N))
        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen] = self.large

        for it in range(self.max_iter):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    L[j - 1, i : i + s] = _f_min_sum(
                        R[j - 1, i : i + s] + L[j, i + s : i + 2 * s],
                        L[j, i : i + s],
                        self.alpha,
                    )
                    L[j - 1, i + s : i + 2 * s] = (
                        _f_min_sum(R[j - 1, i : i + s], L[j, i : i + s], self.alpha)
                        + L[j, i + s : i + 2 * s]
                    )

            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    R[j, i : i + s] = _f_min_sum(
                        R[j - 1, i + s : i + 2 * s] + L[j, i + s : i + 2 * s],
                        R[j - 1, i : i + s],
                        self.alpha,
                    )
                    R[j, i + s : i + 2 * s] = (
                        _f_min_sum(R[j - 1, i : i + s], L[j, i : i + s], self.alpha)
                        + R[j - 1, i + s : i + 2 * s]
                    )

            u_hat = np.zeros(N, dtype=int)
            total = L[0, :] + R[0, :]
            u_hat[total < 0] = 1
            u_hat[self.frozen] = 0

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                return u_hat, it + 1

        u_hat = np.zeros(N, dtype=int)
        total = L[0, :] + R[0, :]
        u_hat[total < 0] = 1
        u_hat[self.frozen] = 0
        return u_hat, self.max_iter
