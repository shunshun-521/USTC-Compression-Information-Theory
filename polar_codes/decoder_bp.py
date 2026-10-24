"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode_core
from decoder_sc import f_operation, _logdomain_sum


class BPDecoder:
    """BP 译码器（极化因子图，min-sum + 早停）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def _f_box(self, a, b):
        if np.isinf(a) and np.isinf(b):
            return np.inf
        return _logdomain_sum(a + b, 0.0) - _logdomain_sum(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        for it in range(self.max_iter):
            num_iters = it + 1
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    L[i, j - 1] = self._f_box(
                        R[i, j] + L[i + s, j], L[i, j]
                    )
                    L[i + s, j - 1] = self._f_box(R[i, j], L[i, j]) + L[i + s, j]

            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    R[i, j] = self._f_box(
                        R[i + s, j - 1] + L[i + s, j], R[i, j - 1]
                    )
                    R[i + s, j] = self._f_box(R[i, j - 1], L[i, j]) + R[i + s, j - 1]

            u_hat = np.zeros(N, dtype=np.int8)
            total = L[:, 0] + R[:, 0]
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total[i] >= 0 else 1

            v_hat = polar_encode_core(u_hat)
            hard = (llr_ch < 0).astype(np.int8)
            if np.array_equal(v_hat, hard):
                break

        u_hat = np.zeros(N, dtype=np.int8)
        total = L[:, 0] + R[:, 0]
        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total[i] >= 0 else 1
        return u_hat, num_iters
