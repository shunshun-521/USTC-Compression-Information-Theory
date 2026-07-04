"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        from encoder import bit_reversal_permutation

        rev = bit_reversal_permutation(self.N)
        llr_ch = llr_ch[rev]
        n = self.n
        N = self.N

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)

        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_bits] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            # 右到左：更新 L 消息
            for col in range(n - 1, -1, -1):
                span = 1 << col
                for j in range(0, N, 2 * span):
                    for k in range(span):
                        idx = j + k
                        L[col, idx] = self._f_min_sum(
                            L[col + 1, idx],
                            L[col + 1, idx + span] + R[col, idx + span],
                        )
                        L[col, idx + span] = (
                            self._f_min_sum(R[col, idx], L[col + 1, idx])
                            + L[col + 1, idx + span]
                        )

            # 左到右：更新 R 消息
            for col in range(0, n):
                span = 1 << col
                for j in range(0, N, 2 * span):
                    for k in range(span):
                        idx = j + k
                        R[col + 1, idx] = self._f_min_sum(
                            R[col, idx],
                            L[col + 1, idx + span] + R[col, idx + span],
                        )
                        R[col + 1, idx + span] = (
                            self._f_min_sum(R[col, idx], L[col + 1, idx])
                            + R[col, idx + span]
                        )

            for i in range(N):
                total = L[0, i] + R[0, i]
                u_hat[i] = 0 if total >= 0 else 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            total = L[0, i] + R[0, i]
            u_hat[i] = 0 if total >= 0 else 1
        u_hat[self.frozen_bits] = 0

        return u_hat, num_iters
