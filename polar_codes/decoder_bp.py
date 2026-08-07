"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


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

    def _minsum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """
        主译码函数。
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(self.max_iter):
            num_iters = it + 1

            for j in range(n, 0, -1):
                stride = 1 << (j - 1)
                for block in range(0, N, stride << 1):
                    left = block
                    right = block + stride
                    L[left:right, j - 1] = self._minsum(
                        R[left:right, j] + L[right:right + stride, j],
                        L[left:right, j],
                    )
                    L[right:right + stride, j - 1] = (
                        self._minsum(R[left:right, j], L[left:right, j])
                        + L[right:right + stride, j]
                    )

            for j in range(0, n):
                stride = 1 << j
                for block in range(0, N, stride << 1):
                    left = block
                    right = block + stride
                    R[left:right, j + 1] = self._minsum(
                        R[right:right + stride, j] + L[right:right + stride, j + 1],
                        R[left:right, j],
                    )
                    R[right:right + stride, j + 1] = (
                        self._minsum(R[left:right, j], L[left:right, j + 1])
                        + R[right:right + stride, j]
                    )

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            u_hat[i] = 0 if total >= 0 else 1
        u_hat[self.frozen_bits] = 0

        return u_hat, num_iters
