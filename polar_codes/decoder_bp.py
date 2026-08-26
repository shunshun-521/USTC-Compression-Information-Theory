"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode, bit_reversal_permutation


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _f_min_sum(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        br = bit_reversal_permutation(N)
        for i in range(N):
            L[i, n] = llr_ch[br[i]]

        for i in range(N):
            if self.frozen_bits[i]:
                R[i, 0] = self.large
            else:
                R[i, 0] = 0.0

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                stride = 1 << (j - 1)
                for block in range(0, N, 2 * stride):
                    for t in range(stride):
                        i = block + t
                        ip = i + stride
                        L[i, j - 1] = self._f_min_sum(
                            R[i, j] + L[ip, j], L[i, j]
                        )
                        L[ip, j - 1] = self._f_min_sum(R[i, j], L[i, j]) + L[ip, j]

            for j in range(1, n + 1):
                stride = 1 << (j - 1)
                for block in range(0, N, 2 * stride):
                    for t in range(stride):
                        i = block + t
                        ip = i + stride
                        R[i, j] = self._f_min_sum(
                            R[ip, j] + L[ip, j], R[i, j - 1]
                        )
                        R[ip, j] = self._f_min_sum(R[i, j - 1], L[i, j]) + R[ip, j]

            u_hat = np.zeros(N, dtype=np.int8)
            for i in range(N):
                total = L[i, 0] + R[i, 0]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
        else:
            u_hat = np.zeros(N, dtype=np.int8)
            for i in range(N):
                total = L[i, 0] + R[i, 0]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1
            num_iters = self.max_iter

        return u_hat, num_iters
