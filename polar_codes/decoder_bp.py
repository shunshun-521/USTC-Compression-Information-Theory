"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation, _frozen_set_from_mask


class BPDecoder:
    """BP 译码器（层状因子图）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_set = _frozen_set_from_mask(frozen_bits)
        self._large = 1e7
        self.rev = bit_reversal_permutation(N)

    def _f_minsum(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        # layer l: 0..n, layer n holds channel observations
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch[self.rev]

        for idx in self.frozen_set:
            R[idx, 0] = self._large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            # L messages: right to left
            for layer in range(n, 0, -1):
                step = 1 << (layer - 1)
                for block in range(0, N, 2 * step):
                    for i in range(step):
                        a = block + i
                        b = a + step
                        L[a, layer - 1] = self._f_minsum(
                            R[a, layer] + L[b, layer], L[a, layer]
                        )
                        L[b, layer - 1] = self._f_minsum(R[a, layer], L[a, layer]) + L[b, layer]

            # R messages: left to right
            for layer in range(1, n + 1):
                step = 1 << (layer - 1)
                for block in range(0, N, 2 * step):
                    for i in range(step):
                        a = block + i
                        b = a + step
                        R[a, layer] = self._f_minsum(
                            R[b, layer] + L[b, layer], R[a, layer - 1]
                        )
                        R[b, layer] = self._f_minsum(R[a, layer - 1], L[a, layer]) + R[b, layer]

            for i in range(N):
                if i in self.frozen_set:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        for i in range(N):
            if i in self.frozen_set:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
