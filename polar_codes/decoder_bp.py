"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import _sign_llr
from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * _sign_llr(a) * _sign_llr(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（因子图列 0..n）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L_msg = np.zeros((N, n + 1), dtype=np.float64)
        R_msg = np.zeros((N, n + 1), dtype=np.float64)
        L_msg[:, n] = llr_ch
        R_msg[:, 0] = 0.0
        R_msg[self.frozen, 0] = self.large

        for num_iter in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    L_msg[i, j - 1] = _f_min_sum(
                        R_msg[i, j] + L_msg[i + s, j],
                        L_msg[i, j],
                        self.alpha,
                    )
                    L_msg[i + s, j - 1] = _f_min_sum(
                        R_msg[i, j], L_msg[i, j], self.alpha
                    ) + L_msg[i + s, j]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    R_msg[i, j + 1] = _f_min_sum(
                        R_msg[i + s, j] + L_msg[i + s, j + 1],
                        R_msg[i, j],
                        self.alpha,
                    )
                    R_msg[i + s, j + 1] = _f_min_sum(
                        R_msg[i, j], L_msg[i, j + 1], self.alpha
                    ) + R_msg[i + s, j]

            u_hat = np.zeros(N, dtype=int)
            total = L_msg[:, 0] + R_msg[:, 0]
            u_hat[total >= 0] = 0
            u_hat[total < 0] = 1
            u_hat[self.frozen] = 0

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                return u_hat, num_iter

        u_hat = np.zeros(N, dtype=int)
        total = L_msg[:, 0] + R_msg[:, 0]
        u_hat[total >= 0] = 0
        u_hat[total < 0] = 1
        u_hat[self.frozen] = 0
        return u_hat, self.max_iter
