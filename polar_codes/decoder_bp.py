"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation
from channel import hard_decision_llr


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6

    def _f_ms(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        """主译码函数"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_natural = llr_ch.copy()
        rev = bit_reversal_permutation(self.N)
        llr_ch = llr_ch[rev]

        n, N = self.n, self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self._large
        R[:, n] = 0.0

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int32)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            # 从右到左更新 L 消息（列 n -> 1）
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        u = i + k
                        l = i + k + s
                        L[u, j - 1] = self._f_ms(R[u, j] + L[l, j], L[u, j])
                        L[l, j - 1] = self._f_ms(R[u, j], L[u, j]) + L[l, j]

            # 从左到右更新 R 消息（列 0 -> n-1）
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        u = i + k
                        l = i + k + s
                        R[u, j + 1] = self._f_ms(R[l, j + 1] + L[l, j + 1], R[u, j])
                        R[l, j + 1] = self._f_ms(R[u, j], L[u, j + 1]) + R[l, j + 1]

            total = L[:, 0] + R[:, 0]
            u_hat = np.where(total >= 0, 0, 1).astype(np.int32)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_natural)
            if np.array_equal(x_hat, x_hard):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.where(total >= 0, 0, 1).astype(np.int32)
        u_hat[self.frozen_bits] = 0
        return u_hat.astype(np.int32), num_iters
