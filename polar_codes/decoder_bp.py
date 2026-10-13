"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器（flooding schedule）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        assert 2 ** self.n == N
        from decoder_sc import _frozen_for_scd

        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_scd = _frozen_for_scd(self.frozen_bits, N)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_scd, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            # 右到左更新 L
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        i0 = i + t
                        i1 = i + t + s
                        L[i0, j - 1] = self._f_ms(
                            R[i0, j] + L[i1, j], L[i0, j]
                        )
                        L[i1, j - 1] = self._f_ms(R[i0, j], L[i0, j]) + L[i1, j]

            # 左到右更新 R
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        i0 = i + t
                        i1 = i + t + s
                        R[i0, j + 1] = self._f_ms(
                            R[i1, j] + L[i1, j + 1], R[i0, j]
                        )
                        R[i1, j + 1] = self._f_ms(R[i0, j], L[i0, j + 1]) + R[i1, j]

            # 判决与早停
            total = L[:, 0] + R[:, 0]
            for i in range(N):
                if self.frozen_scd[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total[i] >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        for i in range(N):
            if self.frozen_scd[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total[i] >= 0 else 1

        return u_hat, num_iters
