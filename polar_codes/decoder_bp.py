"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    """

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        if isinstance(frozen_bits, np.ndarray) and frozen_bits.dtype != bool:
            self.frozen_bits = frozen_bits.astype(bool)
        else:
            self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_ch
        R[0, self.frozen_bits] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for s in range(n - 1, -1, -1):
                step = 1 << s
                for base in range(0, N, 2 * step):
                    for i in range(step):
                        a = base + i
                        b = base + i + step
                        L[s, a] = self._f_min_sum(
                            L[s + 1, a] + R[s + 1, a], L[s + 1, b]
                        )
                        L[s, b] = self._f_min_sum(
                            R[s + 1, a], L[s + 1, a]
                        ) + L[s + 1, b]

            for s in range(0, n):
                step = 1 << s
                for base in range(0, N, 2 * step):
                    for i in range(step):
                        a = base + i
                        b = base + i + step
                        R[s + 1, a] = self._f_min_sum(
                            R[s, b] + L[s + 1, b], R[s, a]
                        )
                        R[s + 1, b] = self._f_min_sum(
                            R[s, a], L[s + 1, a]
                        ) + R[s, b]

            L[n, :] = llr_ch
            R[0, self.frozen_bits] = self.LARGE

            for i in range(N):
                total = L[0, i] + R[0, i]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                return u_hat, num_iters

            num_iters = it

        for i in range(N):
            total = L[0, i] + R[0, i]
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
