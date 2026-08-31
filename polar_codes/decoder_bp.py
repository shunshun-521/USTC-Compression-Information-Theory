"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
参考：Reduced-Complexity BP Decoding for Polar Codes (Yonsei)
"""
import numpy as np
import math
from decoder_sc import f_operation
from encoder import polar_encode
from channel import hard_decision_llr


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e7

    def _f_ms(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        # L[i,j] 和 R[i,j]，j=1..n+1（1-indexed 风格，内部用 0..n）
        # j=0 为信源端，j=n 为信道端
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        for i in self.frozen_set:
            R[i, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # 右到左：更新 L[:, j-1] 从 L[:, j] 和 R[:, j-1]
            # 对 stage j = n, n-1, ..., 1（1-indexed stage）
            for stage in range(n, 0, -1):
                stride = 1 << (stage - 1)
                for i in range(0, N, 2 * stride):
                    for k in range(stride):
                        a = i + k
                        b = i + k + stride
                        col = stage - 1  # 输出列
                        col_in = stage   # 输入列
                        L[a, col] = self._f_ms(
                            L[a, col_in], L[b, col_in] + R[b, col]
                        )
                        L[b, col] = self._f_ms(R[a, col], L[a, col_in]) + L[b, col_in]

            # 左到右：更新 R[:, j] 从 R[:, j-1] 和 L[:, j]
            for stage in range(1, n + 1):
                stride = 1 << (stage - 1)
                for i in range(0, N, 2 * stride):
                    for k in range(stride):
                        a = i + k
                        b = i + k + stride
                        col = stage
                        col_in = stage - 1
                        R[a, col] = self._f_ms(
                            R[b, col_in], L[b, col] + R[b, col_in]
                        )
                        R[b, col] = self._f_ms(R[a, col_in], L[a, col]) + R[b, col_in]

            for i in range(N):
                u_hat[i] = 0 if i in self.frozen_set else (0 if (L[i, 0] + R[i, 0]) >= 0 else 1)

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_decision_llr(llr_ch)):
                num_iters = it
                break

        return u_hat, num_iters
