"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from decoder_sc import f_operation, _channel_llr_to_decoder
from encoder import polar_encode
from channel import hard_decision_llr

LARGE = 1e6


class BPDecoder:
    """BP 译码器（因子图 n+1 列）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_nat = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = _channel_llr_to_decoder(llr_nat)
        N = self.N
        n = self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            # 右到左更新 L
            for j in range(n, 0, -1):
                step = 2 ** (j - 1)
                for i in range(0, N, 2 * step):
                    L[i, j - 1] = self._f_min_sum(
                        R[i, j] + L[i + step, j], L[i, j]
                    )
                    L[i + step, j - 1] = (
                        self._f_min_sum(R[i, j], L[i, j])
                        + L[i + step, j]
                    )

            # 左到右更新 R
            for j in range(1, n + 1):
                step = 2 ** (j - 1)
                for i in range(0, N, 2 * step):
                    R[i, j] = self._f_min_sum(
                        R[i + step, j] + L[i + step, j], R[i, j - 1]
                    )
                    R[i + step, j] = (
                        self._f_min_sum(R[i, j - 1], L[i, j])
                        + R[i + step, j]
                    )

            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    total = L[i, 0] + R[i, 0]
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_nat)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
