"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode
from decoder_sc import _map_channel_llr, f_operation


class BPDecoder:
    """
    BP 译码器。
    """

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def _minsum_f(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        llr_orig = np.asarray(llr_ch, dtype=np.float64)
        llr = _map_channel_llr(llr_orig)
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            for layer in range(n - 1, -1, -1):
                step = 1 << layer
                span = 1 << (layer + 1)
                for block in range(0, N, span):
                    for k in range(step):
                        left = block + k
                        right = block + k + step
                        L[left, layer] = self._minsum_f(
                            R[left, layer + 1] + L[right, layer + 1], L[left, layer + 1]
                        )
                        L[right, layer] = (
                            self._minsum_f(R[left, layer + 1], L[left, layer + 1])
                            + L[right, layer + 1]
                        )

            for layer in range(n):
                step = 1 << layer
                span = 1 << (layer + 1)
                for block in range(0, N, span):
                    for k in range(step):
                        left = block + k
                        right = block + k + step
                        R[left, layer + 1] = self._minsum_f(
                            R[right, layer] + L[right, layer + 1], R[left, layer]
                        )
                        R[right, layer + 1] = (
                            self._minsum_f(R[left, layer], L[left, layer + 1])
                            + R[right, layer]
                        )

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_orig < 0).astype(np.int8)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
