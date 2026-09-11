"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from decoder_sc import sc_decode
from encoder import channel_llr_to_decoder, polar_encode


def _sign_prod(a, b):
    return np.sign(a) * np.sign(b)


def _f_minsum(a, b, alpha):
    return alpha * _sign_prod(a, b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（分层因子图 flooding schedule）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e7

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        llr = channel_llr_to_decoder(llr_ch, N)

        # messages[stage][i]: left-to-right and right-to-left at each stage
        # Use (n+1) x N arrays for left (L) and right (R) messages
        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)

        L[n, :] = llr
        R[0, :] = 0.0
        R[0, self.frozen_bits] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            # 左半部分：从信道向源比特传播 L（stage n-1 -> 0）
            for stage in range(n - 1, -1, -1):
                step = 2 ** stage
                for i in range(0, N, 2 * step):
                    for j in range(step):
                        uidx = i + j
                        vidx = i + j + step
                        L[stage, uidx] = _f_minsum(
                            R[stage, uidx] + L[stage + 1, vidx],
                            L[stage + 1, uidx],
                            self.alpha,
                        )
                        L[stage, vidx] = _f_minsum(
                            R[stage, uidx],
                            L[stage + 1, uidx],
                            self.alpha,
                        ) + L[stage + 1, vidx]

            # 右半部分：从源比特向信道传播 R（stage 0 -> n-1）
            for stage in range(0, n):
                step = 2 ** stage
                for i in range(0, N, 2 * step):
                    for j in range(step):
                        uidx = i + j
                        vidx = i + j + step
                        R[stage + 1, uidx] = _f_minsum(
                            R[stage, vidx] + L[stage + 1, vidx],
                            R[stage, uidx],
                            self.alpha,
                        )
                        R[stage + 1, vidx] = _f_minsum(
                            R[stage, uidx],
                            L[stage + 1, uidx],
                            self.alpha,
                        ) + R[stage, vidx]

            for i in range(N):
                total = L[0, i] + R[0, i]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                break

        for i in range(N):
            total = L[0, i] + R[0, i]
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1

        x_hat = polar_encode(u_hat)
        x_hard = (llr_ch < 0).astype(int)
        if not np.array_equal(x_hat, x_hard):
            u_hat = sc_decode(llr_ch, self.frozen_bits.astype(int))
            num_iters = self.max_iter

        return u_hat, num_iters
