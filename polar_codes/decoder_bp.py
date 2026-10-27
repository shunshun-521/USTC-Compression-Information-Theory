"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode
from decoder_sc import f_operation


def _ms_f(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（极化码因子图）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.info_idx = np.where(self.frozen_bits == 0)[0]
        self.LARGE = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        # L[i][j]: left messages, R[i][j]: right messages
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            # Right to left: update L[:, j] for j = n-1 .. 0
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        Li = i + k
                        Ri = i + k + s
                        L[Li, j] = _ms_f(
                            R[Li, j + 1] + L[Ri, j + 1], L[Li, j + 1], self.alpha
                        )
                        L[Ri, j] = _ms_f(R[Li, j + 1], L[Li, j + 1], self.alpha) + L[Ri, j + 1]

            # Left to right: update R[:, j+1] for j = 0 .. n-1
            for j in range(n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        Li = i + k
                        Ri = i + k + s
                        R[Li, j + 1] = _ms_f(
                            R[Ri, j] + L[Ri, j + 1], R[Li, j], self.alpha
                        )
                        R[Ri, j + 1] = _ms_f(R[Li, j], L[Li, j + 1], self.alpha) + R[Ri, j]

            u_hat = np.zeros(N, dtype=int)
            total = L[:, 0] + R[:, 0]
            u_hat[self.info_idx] = (total[self.info_idx] < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                num_iters = it
                break

        u_hat = np.zeros(N, dtype=int)
        total = L[:, 0] + R[:, 0]
        u_hat[self.info_idx] = (total[self.info_idx] < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
