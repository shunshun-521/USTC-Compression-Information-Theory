"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停
"""
import numpy as np

from decoder_sc import _sign_llr
from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * _sign_llr(a) * _sign_llr(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（分层 L/R 消息，与极化因子图对应）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e8

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen, 0] = self.large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for phi in range(n - 1, -1, -1):
                s = 1 << phi
                for i in range(0, N, 2 * s):
                    R[i, phi] = _f_min_sum(
                        R[i, phi + 1] + L[i + s, phi + 1], R[i + s, phi], self.alpha
                    )
                    R[i + s, phi] = (
                        _f_min_sum(R[i, phi], L[i, phi + 1], self.alpha)
                        + R[i + s, phi + 1]
                    )

            for phi in range(n):
                s = 1 << phi
                for i in range(0, N, 2 * s):
                    L[i, phi + 1] = _f_min_sum(
                        L[i, phi], L[i + s, phi + 1] + R[i, phi], self.alpha
                    )
                    L[i + s, phi + 1] = (
                        _f_min_sum(R[i, phi], L[i, phi + 1], self.alpha)
                        + L[i + s, phi + 1]
                    )

            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[total >= 0] = 0
            u_hat[total < 0] = 1
            u_hat[self.frozen] = 0

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                return u_hat, num_iters

        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[total >= 0] = 0
        u_hat[total < 0] = 1
        u_hat[self.frozen] = 0
        return u_hat, num_iters
