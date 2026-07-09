"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode
from decoder_sc import f_operation, _align_channel_llrs

LARGE = 1e6


class BPDecoder:
    """BP 译码器（因子图 min-sum，含早停）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def _f_ms(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch, bit_reversed_codeword=True):
        llr_ch = _align_channel_llrs(llr_ch, bit_reversed_codeword)
        n = self.n
        N = self.N

        L = [np.zeros(N, dtype=np.float64) for _ in range(n + 1)]
        R = [np.zeros(N, dtype=np.float64) for _ in range(n + 1)]
        L[n][:] = llr_ch
        R[0][:] = 0.0
        R[0][self.frozen_bits] = LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for stage in range(n):
                step = 1 << stage
                for i in range(0, N, 2 * step):
                    for j in range(step):
                        a, b = i + j, i + j + step
                        R[stage + 1][b] = self._f_ms(
                            R[stage + 1][b] + L[stage + 1][b], R[stage][a]
                        )
                        R[stage + 1][a] = self._f_ms(R[stage][a], L[stage + 1][a]) + R[stage + 1][b]

            for stage in range(n - 1, -1, -1):
                step = 1 << stage
                for i in range(0, N, 2 * step):
                    for j in range(step):
                        a, b = i + j, i + j + step
                        L[stage][a] = self._f_ms(L[stage + 1][a], L[stage + 1][b] + R[stage + 1][b])
                        L[stage][b] = self._f_ms(L[stage + 1][a], R[stage + 1][a]) + L[stage + 1][b]

            for i in range(N):
                total = L[0][i] + R[0][i]
                u_hat[i] = 0 if self.frozen_bits[i] or total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
        else:
            num_iters = self.max_iter
            for i in range(N):
                total = L[0][i] + R[0][i]
                u_hat[i] = 0 if self.frozen_bits[i] or total >= 0 else 1

        return u_hat, num_iters
