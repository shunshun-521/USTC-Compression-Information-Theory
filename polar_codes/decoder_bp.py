"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import _prepare_llr, f_operation, g_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _scaled_f(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        llr_raw = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = _prepare_llr(llr_raw)
        N, n = self.N, self.n
        alpha = self.alpha

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_bits] = self.large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for layer in range(n - 1, -1, -1):
                step = 1 << layer
                for i in range(0, N, 2 * step):
                    for j in range(step):
                        idx = i + j
                        L[layer, idx] = self._scaled_f(
                            R[layer, idx] + L[layer + 1, idx + step], L[layer + 1, idx]
                        )
                        L[layer, idx + step] = self._scaled_f(
                            R[layer, idx], L[layer + 1, idx]
                        ) + L[layer + 1, idx + step]

            for layer in range(n):
                step = 1 << layer
                for i in range(0, N, 2 * step):
                    for j in range(step):
                        idx = i + j
                        R[layer + 1, idx] = self._scaled_f(
                            R[layer, idx + step] + L[layer + 1, idx + step], R[layer, idx]
                        )
                        R[layer + 1, idx + step] = self._scaled_f(
                            R[layer, idx], L[layer + 1, idx]
                        ) + R[layer, idx + step]

            for i in range(N):
                total = L[0, i] + R[0, i]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_raw < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        for i in range(N):
            total = L[0, i] + R[0, i]
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat.astype(int), num_iters
