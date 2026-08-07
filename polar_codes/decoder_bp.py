"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math

from encoder import polar_encode
from decoder_sc import f_operation, _frozen_to_set


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_set = _frozen_to_set(frozen_bits)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """
        主译码函数。

        返回：
            u_hat: 估计源序列
            num_iters: 实际迭代次数
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        for idx in self.frozen_set:
            R[idx, 0] = self.LARGE

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            # 右到左更新 L
            for j in range(n - 1, -1, -1):
                stride = 1 << j
                for i in range(0, N, 2 * stride):
                    for k in range(i, i + stride):
                        L[k, j] = self._f_ms(
                            R[k, j] + L[k + stride, j + 1], L[k, j + 1]
                        )
                        L[k + stride, j] = self._f_ms(
                            R[k, j], L[k, j + 1]
                        ) + L[k + stride, j + 1]

            # 左到右更新 R
            for j in range(1, n + 1):
                stride = 1 << (j - 1)
                for i in range(0, N, 2 * stride):
                    for k in range(i, i + stride):
                        R[k, j] = self._f_ms(
                            R[k + stride, j] + L[k + stride, j], R[k, j - 1]
                        )
                        R[k + stride, j] = self._f_ms(
                            R[k, j - 1], L[k, j]
                        ) + R[k + stride, j]

            # 硬判决与早停
            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                if i in self.frozen_set:
                    u_hat[i] = 0
                else:
                    total = L[i, 0] + R[i, 0]
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
        else:
            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                if i in self.frozen_set:
                    u_hat[i] = 0
                else:
                    total = L[i, 0] + R[i, 0]
                    u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
