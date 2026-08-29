"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（按极化码因子图分层消息传递）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6
        self.ibr = bit_reversal_permutation(N)

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_raw = np.asarray(llr_ch, dtype=np.float64)
        y = llr_raw[self.ibr]

        n = self.n
        N = self.N

        left = np.zeros((n + 1, N))
        right = np.zeros((n + 1, N))
        left[n, :] = y
        right[0, :] = 0.0
        right[0, self.frozen_bits] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for s in range(n - 1, -1, -1):
                step = 1 << s
                for i in range(0, N, 2 * step):
                    left[s, i:i + step] = self._f_min_sum(
                        right[s, i:i + step] + left[s + 1, i + step:i + 2 * step],
                        left[s + 1, i:i + step],
                    )
                    left[s, i + step:i + 2 * step] = self._f_min_sum(
                        right[s, i:i + step], left[s + 1, i:i + step]
                    ) + left[s + 1, i + step:i + 2 * step]

            for s in range(0, n):
                step = 1 << s
                for i in range(0, N, 2 * step):
                    right[s + 1, i:i + step] = self._f_min_sum(
                        right[s, i + step:i + 2 * step] + left[s + 1, i + step:i + 2 * step],
                        right[s, i:i + step],
                    )
                    right[s + 1, i + step:i + 2 * step] = self._f_min_sum(
                        right[s, i:i + step], left[s + 1, i:i + step]
                    ) + right[s, i + step:i + 2 * step]

            num_iters = it
            total = left[0, :] + right[0, :]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            if np.array_equal(polar_encode(u_hat), (llr_raw < 0).astype(int)):
                break

        total = left[0, :] + right[0, :]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0

        return u_hat, num_iters
