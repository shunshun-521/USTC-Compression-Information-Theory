"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode
from decoder_sc import _channel_llr_to_decoder


def _f_minsum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（列 0..n，行 0..N-1）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_raw = np.asarray(llr_ch, dtype=np.float64)
        llr_dec = _channel_llr_to_decoder(llr_raw)
        n = self.n
        N = self.N
        LARGE = 1e6

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_dec
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = LARGE

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    L[i, j] = _f_minsum(
                        R[i, j + 1] + L[i + s, j + 1], L[i, j + 1], self.alpha
                    )
                    L[i + s, j] = _f_minsum(R[i, j + 1], L[i, j + 1], self.alpha) + L[
                        i + s, j + 1
                    ]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    R[i, j + 1] = _f_minsum(
                        R[i + s, j] + L[i + s, j + 1], R[i, j], self.alpha
                    )
                    R[i + s, j + 1] = _f_minsum(R[i, j], L[i, j + 1], self.alpha) + R[
                        i + s, j
                    ]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_raw < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
