"""
极化码 BP（置信传播）译码器
基于极化因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode


def _ms_f(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """极化码因子图 BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _hard_bits_from_llr(self, llr_ch):
        return (llr_ch < 0).astype(int)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        Z = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        Z[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = 0
        for _ in range(self.max_iter):
            num_iters += 1

            for layer in range(n):
                step = 1 << layer
                out_col = n - 1 - layer
                in_col = n - layer
                for t in range(0, N, 2 * step):
                    j = t + step
                    Z[t, out_col] = _ms_f(
                        Z[t, in_col], R[j, out_col] + Z[j, in_col], self.alpha
                    )
                    Z[j, out_col] = _ms_f(Z[t, in_col], R[t, out_col], self.alpha) + Z[
                        j, in_col
                    ]

            for layer in range(n):
                step = 1 << layer
                z_col = n - layer
                for t in range(0, N, 2 * step):
                    j = t + step
                    R[t, layer + 1] = _ms_f(
                        R[t, layer], Z[t, z_col] + R[j, layer], self.alpha
                    )
                    R[j, layer + 1] = _ms_f(R[t, layer], Z[t, z_col], self.alpha) + R[
                        j, layer
                    ]

            total = Z[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, self._hard_bits_from_llr(llr_ch)):
                break

        total = Z[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
