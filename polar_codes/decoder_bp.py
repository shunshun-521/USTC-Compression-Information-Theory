"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation, _prepare_llr, _bit_reversal_array


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 20.0

    def _boxplus(self, a, b):
        a = np.clip(a, -self.large, self.large)
        b = np.clip(b, -self.large, self.large)
        return 2.0 * np.arctanh(np.tanh(a / 2.0) * np.tanh(b / 2.0))

    def _f_min_sum(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = _prepare_llr(llr_ch)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = np.clip(llr_ch, -self.large, self.large)
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = self.max_iter
        rev = _bit_reversal_array(N)
        g = self._boxplus

        for it in range(1, self.max_iter + 1):
            for s in range(n):
                block = 2 ** s
                for b in range(0, N, 2 * block):
                    for k in range(block):
                        i = b + k
                        j = i + block
                        R[i, s + 1] = g(R[i, s], L[j, s + 1] + R[j, s + 1])
                        R[j, s + 1] = g(R[i, s], L[i, s + 1]) + R[j, s]

            for s in range(n - 1, -1, -1):
                block = 2 ** s
                for b in range(0, N, 2 * block):
                    for k in range(block):
                        i = b + k
                        j = i + block
                        L[i, s] = g(L[i, s + 1], L[j, s + 1] + R[j, s + 1])
                        L[j, s] = g(R[i, s + 1], L[i, s + 1]) + L[j, s + 1]

            total = L[:, 0] + R[:, 0]
            u_hat = np.zeros(N, dtype=int)
            u_hat[total < 0] = 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat[rev], hard_ch):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(N, dtype=int)
        u_hat[total < 0] = 1
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
