"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（min-sum + 早停）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        for i in range(N):
            if self.frozen_bits[i]:
                R[i, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            for layer in range(n, 0, -1):
                s = 1 << (layer - 1)
                for i in range(0, N, 2 * s):
                    for j in range(s):
                        idx = i + j
                        idx2 = idx + s
                        L[idx, layer - 1] = self._f_min_sum(
                            R[idx, layer - 1] + L[idx2, layer],
                            L[idx, layer],
                        )
                        L[idx2, layer - 1] = (
                            self._f_min_sum(R[idx, layer - 1], L[idx, layer])
                            + L[idx2, layer]
                        )

            for layer in range(0, n):
                s = 1 << layer
                for i in range(0, N, 2 * s):
                    for j in range(s):
                        idx = i + j
                        idx2 = idx + s
                        R[idx, layer + 1] = self._f_min_sum(
                            R[idx2, layer] + L[idx2, layer + 1],
                            R[idx, layer],
                        )
                        R[idx2, layer + 1] = (
                            self._f_min_sum(R[idx, layer], L[idx, layer + 1])
                            + R[idx2, layer]
                        )

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    total = L[i, 0] + R[i, 0]
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
