"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import sc_decode, f_operation


class BPDecoder:
    """BP 译码器（min-sum + 早停）；与极化因子图一致的简化消息传递。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _f_ms(self, a, b):
        return self.alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

    def decode(self, llr_ch):
        n = self.n
        N = self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        y = np.asarray(llr_ch, dtype=np.float64)
        L[:, n] = y
        R[:, 0] = 0.0
        R[self.frozen, 0] = self.large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    L[i, j - 1] = self._f_ms(R[i, j] + L[i + s, j], L[i, j])
                    L[i + s, j - 1] = self._f_ms(R[i, j], L[i, j]) + L[i + s, j]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    R[i, j + 1] = self._f_ms(R[i + s, j] + L[i + s, j + 1], R[i, j])
                    R[i + s, j + 1] = self._f_ms(R[i, j], L[i, j + 1]) + R[i + s, j]

            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                if self.frozen[i]:
                    u_hat[i] = 0
                else:
                    total = L[i, 0] + R[i, 0]
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard = (y < 0).astype(int)
            if np.array_equal(x_hat, hard):
                num_iters = it
                break
        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            if self.frozen[i]:
                u_hat[i] = 0
            else:
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1
        return u_hat, num_iters
