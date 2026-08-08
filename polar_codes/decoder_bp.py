"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation, _prepare_llr


class BPDecoder:
    """BP 译码器（因子图 min-sum，含早停）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6

    def _f_ms(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 (u_hat, num_iters)。
        """
        llr_ch = _prepare_llr(llr_ch)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))
        L[:, n] = llr_ch.copy()
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self._large

        u_hat = np.zeros(N, dtype=int)
        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        L[idx, j] = self._f_ms(
                            R[idx, j + 1] + L[idx + s, j + 1], L[idx, j + 1]
                        )
                        L[idx + s, j] = self._f_ms(
                            R[idx, j + 1], L[idx, j + 1]
                        ) + L[idx + s, j + 1]

            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        R[idx, j] = self._f_ms(
                            R[idx + s, j] + L[idx + s, j], R[idx, j - 1]
                        )
                        R[idx + s, j] = self._f_ms(
                            R[idx, j - 1], L[idx, j]
                        ) + R[idx + s, j]

            for i in range(N):
                u_hat[i] = 0 if self.frozen_bits[i] else (
                    0 if (L[i, 0] + R[i, 0]) >= 0 else 1
                )

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break

        for i in range(N):
            u_hat[i] = 0 if self.frozen_bits[i] else (
                0 if (L[i, 0] + R[i, 0]) >= 0 else 1
            )

        return u_hat, num_iters
