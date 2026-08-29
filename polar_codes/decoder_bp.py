"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器（stage 0=信源端，stage n=信道端）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_min_sum(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)

        L[n] = llr_ch
        R[0] = 0.0
        for idx in np.where(self.frozen_bits == 1)[0]:
            R[0, idx] = self.LARGE

        num_iters = self.max_iter

        for it in range(self.max_iter):
            # 信道端 → 信源端
            for stage in range(n - 1, -1, -1):
                bs = 2 ** (stage + 1)
                h = 2 ** stage
                for i in range(0, N, bs):
                    L[stage, i:i + h] = self._f_min_sum(
                        R[stage + 1, i:i + h] + L[stage + 1, i + h:i + bs],
                        L[stage + 1, i:i + h]
                    )
                    L[stage, i + h:i + bs] = self._f_min_sum(
                        R[stage + 1, i:i + h],
                        L[stage + 1, i:i + h]
                    ) + L[stage + 1, i + h:i + bs]

            # 信源端 → 信道端
            for stage in range(n):
                bs = 2 ** (stage + 1)
                h = 2 ** stage
                for i in range(0, N, bs):
                    R[stage + 1, i:i + h] = self._f_min_sum(
                        R[stage, i + h:i + bs] + L[stage + 1, i + h:i + bs],
                        R[stage, i:i + h]
                    )
                    R[stage + 1, i + h:i + bs] = self._f_min_sum(
                        R[stage, i:i + h],
                        L[stage + 1, i:i + h]
                    ) + R[stage, i + h:i + bs]

            total = L[0] + R[0]
            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total[i] >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it + 1
                break

        total = L[0] + R[0]
        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total[i] >= 0 else 1

        return u_hat, num_iters
