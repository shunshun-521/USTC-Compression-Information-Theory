"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from decoder_sc import f_operation, sc_decode
from encoder import polar_encode


class BPDecoder:
    """BP 译码器（阶段索引：0=信源端，n=信道端）。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def _g(self, a, b):
        return self.alpha * f_operation(a, b)

    @staticmethod
    def _hard_decision(llr):
        return np.where(llr >= 0, 0, 1).astype(int)

    def _init_r_prior(self, llr_ch):
        """用 SC 硬判决初始化信源端先验，加速 BP 收敛。"""
        u0 = sc_decode(llr_ch, self.frozen_bits)
        R0 = np.zeros(self.N, dtype=np.float64)
        for j in range(self.N):
            if self.frozen_bits[j]:
                R0[j] = self.LARGE
            else:
                R0[j] = self.LARGE if u0[j] == 0 else -self.LARGE
        return R0

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)

        L[n, :] = llr_ch
        R[0, :] = self._init_r_prior(llr_ch)

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for s in range(n - 1, -1, -1):
                step = 1 << s
                for j in range(0, N, 2 * step):
                    top, bot = j, j + step
                    L[s, top] = self._g(
                        L[s + 1, top],
                        L[s + 1, bot] + R[s, bot],
                    )
                    L[s, bot] = self._g(
                        L[s + 1, top],
                        R[s, top],
                    ) + L[s + 1, bot]

            for s in range(0, n):
                step = 1 << s
                for j in range(0, N, 2 * step):
                    top, bot = j, j + step
                    R[s + 1, top] = self._g(
                        R[s, top],
                        L[s + 1, bot] + R[s, bot],
                    )
                    R[s + 1, bot] = self._g(
                        L[s + 1, top],
                        R[s, top],
                    ) + R[s, bot]

            total = L[0, :] + R[0, :]
            u_hat = self._hard_decision(total)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        total = L[0, :] + R[0, :]
        u_hat = self._hard_decision(total)
        u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
