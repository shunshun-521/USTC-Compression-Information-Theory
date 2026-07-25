"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from decoder_sc import f_operation, _sign_ms, bit_reversed, _permute_llr
from encoder import polar_encode


LARGE = 1e6


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def _f_ms(self, a, b):
        return self.alpha * _sign_ms(a) * _sign_ms(b) * np.minimum(np.abs(a), np.abs(b))

    def decode(self, llr_ch):
        """主译码函数。"""
        N, n = self.N, self.n
        llr_ch = _permute_llr(llr_ch, N)

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(self.max_iter):
            num_iters = it + 1

            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx0 = i + k
                        idx1 = i + k + s
                        L[idx0, j - 1] = self._f_ms(
                            R[idx0, j] + L[idx1, j], L[idx0, j]
                        )
                        L[idx1, j - 1] = self._f_ms(R[idx0, j], L[idx0, j]) + L[idx1, j]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx0 = i + k
                        idx1 = i + k + s
                        R[idx0, j + 1] = self._f_ms(
                            R[idx1, j] + L[idx1, j + 1], R[idx0, j]
                        )
                        R[idx1, j + 1] = self._f_ms(R[idx0, j], L[idx0, j + 1]) + R[idx1, j]

            for i in range(N):
                u_hat[i] = 0 if self.frozen_bits[i] else (0 if (L[i, 0] + R[i, 0]) >= 0 else 1)

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            brp = np.array([bit_reversed(k, n) for k in range(N)])
            if np.array_equal(x_hat[brp], hard_ch):
                break

        for i in range(N):
            u_hat[i] = 0 if self.frozen_bits[i] else (0 if (L[i, 0] + R[i, 0]) >= 0 else 1)

        return u_hat, num_iters
