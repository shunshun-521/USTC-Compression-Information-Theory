"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode, bit_reversal_permutation


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        fb = np.asarray(frozen_bits)
        self.frozen_bits = fb.astype(bool) if fb.dtype == bool else (fb != 0)
        self.max_iter = max_iter
        self.alpha = alpha
        self.rev = bit_reversal_permutation(N)
        self.LARGE = 1e6

    def _f_minsum(self, x, y):
        return self.alpha * np.sign(x) * np.sign(y) * min(abs(x), abs(y))

    def decode(self, llr_ch):
        llr_orig = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = llr_orig[self.rev]
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        for i in range(N):
            R[i, 0] = self.LARGE if self.frozen_bits[i] else 0.0

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)
        x_hard = (llr_orig < 0).astype(int)

        for it in range(1, self.max_iter + 1):
            # 右到左更新 L，列 j: n-1 -> 0
            for j in range(n - 1, -1, -1):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        a = i + k
                        b = i + k + step
                        L[a, j] = self._f_minsum(
                            R[a, j + 1] + L[b, j + 1], L[a, j + 1]
                        )
                        L[b, j] = self._f_minsum(R[a, j + 1], L[a, j + 1]) + L[b, j + 1]

            # 左到右更新 R，列 j: 0 -> n-1
            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        a = i + k
                        b = i + k + step
                        R[a, j + 1] = self._f_minsum(
                            R[b, j + 1] + L[b, j + 1], R[a, j]
                        )
                        R[b, j + 1] = self._f_minsum(R[a, j], L[a, j + 1]) + R[b, j]

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            if np.array_equal(polar_encode(u_hat), x_hard):
                num_iters = it
                break
            num_iters = it

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
