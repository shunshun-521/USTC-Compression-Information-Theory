"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import _map_channel_llr
from encoder import polar_encode


def _ms_f(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（极化码因子图，min-sum）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e8

    def decode(self, llr_ch):
        llr_ch = _map_channel_llr(llr_ch)
        n, N = self.n, self.N

        # L[i, s] 与 R[i, s]，s=0 为信源侧，s=n 为信道侧
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for s in range(n - 1, -1, -1):
                block = 1 << (s + 1)
                half = block >> 1
                for base in range(0, N, block):
                    for i in range(half):
                        a = base + i
                        b = base + i + half
                        L[a, s] = _ms_f(R[a, s] + L[b, s + 1], L[a, s + 1], self.alpha)
                        L[b, s] = _ms_f(R[a, s], L[a, s + 1], self.alpha) + L[b, s + 1]

            for s in range(1, n + 1):
                block = 1 << s
                half = block >> 1
                for base in range(0, N, block):
                    for i in range(half):
                        a = base + i
                        b = base + i + half
                        R[a, s] = _ms_f(R[b, s - 1] + L[b, s], R[a, s - 1], self.alpha)
                        R[b, s] = _ms_f(R[a, s - 1], L[a, s], self.alpha) + R[b, s - 1]

            total = L[:, 0] + R[:, 0]
            u_hat = np.where(total >= 0, 0, 1).astype(np.int8)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard_ch):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.where(total >= 0, 0, 1).astype(np.int8)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
