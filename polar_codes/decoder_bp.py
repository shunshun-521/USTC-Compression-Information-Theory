"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_min_sum


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self._large = 1e6

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 (u_hat, num_iters)。
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        alpha = self.alpha

        # L[i][j]: 从右到左消息，R[i][j]: 从左到右消息
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        llr_core = llr_ch[self.br]
        L[:, n] = llr_core
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self._large

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            # 从右到左更新 L
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx = i + k
                        La = R[idx, j - 1] + L[idx + step, j]
                        Lb = L[idx, j]
                        L[idx, j - 1] = f_min_sum(La, Lb, alpha)
                        La2 = R[idx, j - 1]
                        Lb2 = L[idx, j]
                        Rb = L[idx + step, j]
                        L[idx + step, j - 1] = f_min_sum(La2, Lb2, alpha) + Rb

            # 从左到右更新 R
            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx = i + k
                        La = R[idx + step, j] + L[idx + step, j + 1]
                        Lb = R[idx, j - 1] if j > 0 else R[idx, 0]
                        R[idx, j + 1] = f_min_sum(La, Lb, alpha)
                        La2 = R[idx, j - 1] if j > 0 else R[idx, 0]
                        Lb2 = L[idx, j + 1]
                        Rb = L[idx + step, j]
                        R[idx + step, j + 1] = f_min_sum(La2, Lb2, alpha) + Rb

            # 硬判决与早停
            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
        else:
            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

        return u_hat, num_iters
