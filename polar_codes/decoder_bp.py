"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode


class BPDecoder:
    """BP 译码器（阶段索引 0..n，第 n 列为信道 LLR）。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(self.frozen_bits == 0)[0]
        self.max_iter = max_iter
        self.alpha = alpha

    def _g_ms(self, a, b):
        return self.alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

    def decode(self, llr_ch):
        """主译码函数。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_ch
        R[0, self.info_idx] = 0.0
        R[0, self.frozen_idx] = self.LARGE

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for stage in range(n - 1, -1, -1):
                stride = 2 ** stage
                for j in range(0, N, 2 * stride):
                    for k in range(stride):
                        a = j + k
                        b = j + k + stride
                        L[stage, a] = self._g_ms(
                            L[stage + 1, a], L[stage + 1, b] + R[stage, b]
                        )
                        L[stage, b] = (
                            self._g_ms(L[stage + 1, a], R[stage, a])
                            + L[stage + 1, b]
                        )

            for stage in range(0, n):
                stride = 2 ** stage
                for j in range(0, N, 2 * stride):
                    for k in range(stride):
                        a = j + k
                        b = j + k + stride
                        R[stage + 1, a] = self._g_ms(
                            R[stage, a], L[stage + 1, b] + R[stage, b]
                        )
                        R[stage + 1, b] = (
                            self._g_ms(L[stage + 1, a], R[stage, a])
                            + R[stage, b]
                        )

            u_hat = self._hard_decision(L, R)
            if self._early_stop(u_hat, llr_ch):
                num_iters = it
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters

    def _hard_decision(self, L, R):
        total = L[0, :] + R[0, :]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat

    def _early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, hard_ch)
