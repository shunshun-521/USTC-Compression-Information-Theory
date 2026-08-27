"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode, channel_llr_to_decoder


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)"""
        llr_orig = np.asarray(llr_ch, dtype=np.float64)
        llr_internal = channel_llr_to_decoder(llr_orig)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_internal
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    L[i : i + s, j] = self._f_min_sum(
                        R[i : i + s, j + 1] + L[i + s : i + 2 * s, j + 1],
                        L[i : i + s, j + 1],
                    )
                    L[i + s : i + 2 * s, j] = self._f_min_sum(
                        R[i : i + s, j + 1], L[i : i + s, j + 1]
                    ) + L[i + s : i + 2 * s, j + 1]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    R[i : i + s, j + 1] = self._f_min_sum(
                        R[i + s : i + 2 * s, j] + L[i + s : i + 2 * s, j + 1],
                        R[i : i + s, j],
                    )
                    R[i + s : i + 2 * s, j] = self._f_min_sum(
                        R[i : i + s, j], L[i : i + s, j + 1]
                    ) + R[i + s : i + 2 * s, j]

            total_llr = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[total_llr < 0] = 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_orig < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total_llr = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[total_llr < 0] = 1
        u_hat[self.frozen_bits] = 0

        return u_hat, num_iters
