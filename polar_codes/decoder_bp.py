"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode, bit_reversal_permutation


class BPDecoder:
    """BP 译码器（min-sum + 早停）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6
        self._br = bit_reversal_permutation(N)

    def _f_ms(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_tx = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = llr_tx[self._br]
        n, N = self.n, self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self._large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            # 右到左更新 L
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        L[i + k, j - 1] = self._f_ms(
                            R[i + k, j] + L[i + k + s, j], L[i + k, j]
                        )
                        L[i + k + s, j - 1] = self._f_ms(R[i + k, j], L[i + k, j]) + L[
                            i + k + s, j
                        ]

            # 左到右更新 R
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        R[i + k, j + 1] = self._f_ms(
                            R[i + k + s, j] + L[i + k + s, j + 1], R[i + k, j]
                        )
                        R[i + k + s, j + 1] = self._f_ms(R[i + k, j], L[i + k, j + 1]) + R[
                            i + k + s, j
                        ]

            u_hat = np.zeros(N, dtype=int)
            total = L[:, 0] + R[:, 0]
            u_hat[total < 0] = 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_tx < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[total < 0] = 1
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
