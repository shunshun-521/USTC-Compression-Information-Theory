"""
极化码 BP（置信传播）译码器
min-sum 近似 + 早停
"""
import numpy as np
import math
from encoder import polar_encode


class BPDecoder:
    """BP 译码器（因子图 stage 0=比特端, stage n=信道端）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e7

    def _f_ms(self, a, b):
        sa = np.sign(a) * np.sign(b)
        sa = np.where(sa == 0, 1.0, sa)
        return self.alpha * sa * np.minimum(np.abs(a), np.abs(b))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen] = self.large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            # 信道侧 → 信源侧
            for stage in range(n, 0, -1):
                step = 1 << (stage - 1)
                for i in range(0, N, 2 * step):
                    L[stage - 1, i : i + step] = self._f_ms(
                        R[stage, i : i + step] + L[stage, i + step : i + 2 * step],
                        L[stage, i : i + step],
                    )
                    L[stage - 1, i + step : i + 2 * step] = (
                        self._f_ms(R[stage, i : i + step], L[stage, i : i + step])
                        + L[stage, i + step : i + 2 * step]
                    )

            # 信源侧 → 信道侧
            for stage in range(0, n):
                step = 1 << stage
                for i in range(0, N, 2 * step):
                    R[stage + 1, i : i + step] = self._f_ms(
                        R[stage + 1, i + step : i + 2 * step]
                        + L[stage + 1, i + step : i + 2 * step],
                        R[stage, i : i + step],
                    )
                    R[stage + 1, i + step : i + 2 * step] = (
                        self._f_ms(R[stage, i : i + step], L[stage + 1, i : i + step])
                        + R[stage + 1, i + step : i + 2 * step]
                    )

            post = L[0, :] + R[0, :]
            u_hat = np.zeros(N, dtype=int)
            u_hat[post < 0] = 1
            u_hat[self.frozen] = 0
            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break

        post = L[0, :] + R[0, :]
        u_hat = np.zeros(N, dtype=int)
        u_hat[post < 0] = 1
        u_hat[self.frozen] = 0
        return u_hat, num_iters
