"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import align_channel_llr, sc_decode
from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """Layered min-sum BP；若迭代结果不构成合法码字则回退 SC。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def decode(self, llr_ch):
        llr_raw = np.asarray(llr_ch, dtype=np.float64)
        y = align_channel_llr(llr_raw)
        n = self.n
        N = self.N

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = y
        R[0, self.frozen_bits] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)
        x_hard = (llr_raw < 0).astype(int)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            for s in range(n - 1, -1, -1):
                stride = 1 << s
                for phi in range(0, N, stride << 1):
                    L[s, phi] = _f_min_sum(
                        L[s + 1, phi] + R[s + 1, phi + stride],
                        L[s + 1, phi + stride],
                        self.alpha,
                    )
                    L[s, phi + stride] = _f_min_sum(
                        R[s + 1, phi],
                        L[s + 1, phi],
                        self.alpha,
                    ) + L[s + 1, phi + stride]

            for s in range(0, n):
                stride = 1 << s
                for phi in range(0, N, stride << 1):
                    R[s + 1, phi] = _f_min_sum(
                        R[s + 1, phi + stride] + L[s + 1, phi + stride],
                        R[s, phi],
                        self.alpha,
                    )
                    R[s + 1, phi + stride] = _f_min_sum(
                        R[s, phi],
                        L[s + 1, phi],
                        self.alpha,
                    ) + R[s + 1, phi + stride]

            for i in range(N):
                u_hat[i] = 0 if (L[0, i] + R[0, i]) >= 0 else 1
            u_hat[self.frozen_bits] = 0

            if np.array_equal(polar_encode(u_hat), x_hard):
                return u_hat, num_iters

        if not np.array_equal(polar_encode(u_hat), x_hard):
            u_hat = sc_decode(llr_raw, self.frozen_bits)
        return u_hat, num_iters
