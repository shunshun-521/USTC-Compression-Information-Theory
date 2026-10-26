"""
极化码 BP（置信传播）译码器
因子图 min-sum，含早停
"""
import numpy as np

from encoder import polar_encode
from polar_ops import f_operation
from channel import hard_decision_bits


class BPDecoder:
    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # 右到左更新 L
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for block in range(0, N, s << 1):
                    for i in range(s):
                        left = block + i
                        right = block + i + s
                        L[left, j - 1] = self._f_ms(
                            R[left, j] + L[right, j], L[left, j]
                        )
                        L[right, j - 1] = self._f_ms(R[left, j], L[left, j]) + L[right, j]

            # 左到右更新 R
            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for block in range(0, N, s << 1):
                    for i in range(s):
                        left = block + i
                        right = block + i + s
                        R[left, j] = self._f_ms(
                            R[right, j] + L[right, j], R[left, j - 1]
                        )
                        R[right, j] = self._f_ms(R[left, j - 1], L[left, j]) + R[right, j]

            # 判决与早停
            post = L[:, 0] + R[:, 0]
            u_hat = (post < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_decision_bits(llr_ch)):
                num_iters = it
                break

        return u_hat, num_iters
