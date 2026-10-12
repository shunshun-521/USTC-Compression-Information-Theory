"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import channel_llr_to_decoder, polar_encode
from decoder_sc import f_operation


def _boxplus(a, b, alpha, min_sum):
    if min_sum:
        return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))
    return np.logaddexp(0.0, a + b) - np.logaddexp(a, b)


class BPDecoder:
    """BP 译码器（因子图列 0..n）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375, min_sum=True):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.min_sum = min_sum
        self.LARGE = 1e6

    def decode(self, llr_ch):
        llr_raw = np.asarray(llr_ch, dtype=np.float64)
        llr_dec = channel_llr_to_decoder(llr_raw)
        n, N = self.n, self.N

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_dec
        R[0, :] = 0.0
        R[0, self.frozen_bits] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(i, i + s):
                        L[j - 1, k] = _boxplus(
                            R[j, k] + L[j, k + s], L[j, k], self.alpha, self.min_sum
                        )
                        L[j - 1, k + s] = (
                            _boxplus(R[j, k], L[j, k], self.alpha, self.min_sum)
                            + L[j, k + s]
                        )

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(i, i + s):
                        R[j + 1, k] = _boxplus(
                            R[j + 1, k + s] + L[j + 1, k + s], R[j, k], self.alpha, self.min_sum
                        )
                        R[j + 1, k + s] = (
                            _boxplus(R[j, k], L[j + 1, k], self.alpha, self.min_sum)
                            + R[j + 1, k + s]
                        )

            for i in range(N):
                u_hat[i] = 0 if (L[0, i] + R[0, i]) >= 0 else 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_x = (llr_raw < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break

        for i in range(N):
            u_hat[i] = 0 if (L[0, i] + R[0, i]) >= 0 else 1
        u_hat[self.frozen_bits] = 0

        return u_hat.astype(int), num_iters
