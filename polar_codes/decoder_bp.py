"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
参考: MDPI Symmetry 2022 polar BP min-sum 更新规则
"""
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode
from channel import hard_decision_llr


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.LARGE = 100.0

    def _g(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        m = self.n

        # stage i=0 为信源端，i=m 为信道端
        L = np.zeros((m + 1, N), dtype=np.float64)
        R = np.zeros((m + 1, N), dtype=np.float64)

        L[m, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_idx] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for i in range(m - 1, -1, -1):
                stride = 1 << i
                for j in range(0, N - 2 * stride + 1, 2 * stride):
                    L[i, j] = self._g(
                        L[i + 1, j],
                        L[i + 1, j + stride] + R[i, j + stride],
                    )
                    L[i, j + stride] = self._g(L[i + 1, j], R[i, j]) + L[i + 1, j + stride]

            for i in range(0, m):
                stride = 1 << i
                for j in range(0, N - 2 * stride + 1, 2 * stride):
                    R[i + 1, j] = self._g(
                        R[i, j],
                        L[i + 1, j + stride] + R[i, j + stride],
                    )
                    R[i + 1, j + stride] = self._g(L[i + 1, j], R[i, j]) + R[i, j + stride]

            total = L[0, :] + R[0, :]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                break

        total = L[0, :] + R[0, :]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
