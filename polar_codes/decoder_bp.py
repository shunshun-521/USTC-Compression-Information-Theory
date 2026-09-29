"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（自然序 LLR，与 mcba1n 编码配套）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def _f(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.clip(np.asarray(llr_ch, dtype=np.float64), -1e3, 1e3)
        n = self.n
        N = self.N
        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = 1e6

        def hard_from_llr(llr):
            return (llr < 0).astype(int)

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        idx = i + t
                        L[idx, j - 1] = self._f(
                            R[idx, j - 1] + L[idx + s, j], L[idx, j]
                        )
                        L[idx + s, j - 1] = self._f(R[idx, j - 1], L[idx, j]) + L[
                            idx + s, j
                        ]

            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        idx = i + t
                        R[idx, j] = self._f(
                            R[idx + s, j] + L[idx + s, j], R[idx, j - 1]
                        )
                        R[idx + s, j] = self._f(R[idx, j - 1], L[idx, j]) + R[
                            idx + s, j - 1
                        ]

            total = L[:, 0] + R[:, 0]
            u_hat = hard_from_llr(total)
            u_hat[self.frozen_idx] = 0
            x_hat = polar_encode(u_hat)
            x_hard = hard_from_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = hard_from_llr(total)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
