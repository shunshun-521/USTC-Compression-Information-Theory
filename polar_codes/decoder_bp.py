"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode


LARGE = 1e10


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def _gbox(self, a, b):
        return self.alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(self.max_iter):
            num_iters = it + 1

            for j in range(n - 1, -1, -1):
                step = 2 ** j
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        top = i + k
                        bot = top + step
                        L[top, j] = self._gbox(
                            L[top, j + 1],
                            R[top, j] + L[bot, j + 1],
                        )
                        L[bot, j] = self._gbox(L[top, j + 1], R[top, j]) + L[bot, j + 1]

            for j in range(0, n):
                step = 2 ** j
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        top = i + k
                        bot = top + step
                        R[top, j + 1] = self._gbox(
                            R[top, j],
                            L[bot, j + 1] + R[bot, j],
                        )
                        R[bot, j + 1] = self._gbox(R[top, j], L[top, j + 1]) + R[bot, j]

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
