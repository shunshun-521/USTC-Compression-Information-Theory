"""
极化码 BP（置信传播）译码器：因子图 min-sum BP + 早停
"""
import numpy as np

from decoder_sc import _lower_llr, _upper_llr, channel_llr_to_decoder
from encoder import polar_encode


class BPDecoder:
    """BP 译码器（L/R 消息，列 0..n；列 n 为信道观测）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e7

    def _f(self, a, b):
        if self.alpha != 1.0:
            return self.alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))
        return _upper_llr(a, b)

    def _g(self, top, btm, u):
        return _lower_llr(btm, top, u)

    def decode(self, llr_ch, apply_br_reorder=True):
        if apply_br_reorder:
            llr = channel_llr_to_decoder(llr_ch, self.N).astype(np.float64)
        else:
            llr = np.asarray(llr_ch, dtype=np.float64).copy()

        N, n = self.N, self.n
        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            L = np.zeros((N, n + 1))
            R = np.zeros((N, n + 1))
            L[:, n] = llr
            R[:, 0] = 0.0
            R[self.frozen_bits, 0] = self.LARGE

            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        if j == n:
                            La, Lb = L[idx, j], L[idx + s, j]
                            Ra, Rb = R[idx, j], R[idx + s, j]
                        else:
                            La = L[idx, j + 1]
                            Lb = L[idx + s, j + 1]
                            Ra, Rb = R[idx, j], R[idx + s, j]
                        L[idx, j - 1] = self._f(Ra + La, Rb + Lb)
                        L[idx + s, j - 1] = self._f(Ra, La) + Lb + Rb

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        top_bit = 0 if (L[idx, j] + R[idx, j]) >= 0 else 1
                        R[idx, j + 1] = self._g(
                            R[idx + s, j] + L[idx + s, j + 1],
                            R[idx, j],
                            top_bit,
                        )
                        R[idx + s, j + 1] = self._g(
                            R[idx, j],
                            L[idx, j + 1],
                            top_bit,
                        ) + R[idx + s, j]

            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[total < 0] = 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, (llr < 0).astype(int)):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[total < 0] = 1
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
