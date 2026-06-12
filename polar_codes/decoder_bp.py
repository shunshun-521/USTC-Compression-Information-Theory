"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode


def _ms_f(x, y, alpha):
    sx = 1.0 if x >= 0 else -1.0
    sy = 1.0 if y >= 0 else -1.0
    if x == 0:
        sx = 0.0
    if y == 0:
        sy = 0.0
    return alpha * sx * sy * min(abs(x), abs(y))


class BPDecoder:
    """BP 译码器：stage 0 为信源端，stage n 为信道端。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self._large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_idx] = self._large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for stage in range(n):
                block = 1 << (stage + 1)
                half = block >> 1
                for i in range(0, N, block):
                    for j in range(half):
                        a = i + j
                        b = i + j + half
                        R[stage + 1, a] = _ms_f(
                            R[stage, a] + L[stage, b], R[stage, b], self.alpha
                        )
                        R[stage + 1, b] = _ms_f(R[stage, a], L[stage, b], self.alpha) + R[stage, b]

            for stage in range(n, 0, -1):
                block = 1 << stage
                half = block >> 1
                for i in range(0, N, block):
                    for j in range(half):
                        a = i + j
                        b = i + j + half
                        L[stage - 1, a] = _ms_f(
                            R[stage, a] + L[stage, b], L[stage, a], self.alpha
                        )
                        L[stage - 1, b] = _ms_f(R[stage, a], L[stage, a], self.alpha) + L[stage, b]

            for i in range(N):
                u_hat[i] = 0 if self.frozen_bits[i] else (0 if (L[0, i] + R[0, i]) >= 0 else 1)

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        for i in range(N):
            u_hat[i] = 0 if self.frozen_bits[i] else (0 if (L[0, i] + R[0, i]) >= 0 else 1)

        return u_hat, num_iters
