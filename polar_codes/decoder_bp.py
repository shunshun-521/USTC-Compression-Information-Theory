"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e7

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def _g_soft(self, la, lb, u_llr):
        """软 g 运算，u_llr 为左子树比特的软 LLR"""
        p1 = 1.0 / (1.0 + np.exp(u_llr))
        return la * (1.0 - 2.0 * p1) + lb

    def decode(self, llr_ch):
        """
        主译码函数。

        返回：
            u_hat: 长度 N 的估计源序列
            num_iters: 实际迭代次数
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_ch
        u_llr = np.zeros(N, dtype=np.float64)
        u_llr[self.frozen_bits] = self._large

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for layer in range(n - 1, -1, -1):
                step = 1 << layer
                for i in range(0, N, 2 * step):
                    L[layer, i] = self._f_ms(L[layer + 1, i], L[layer + 1, i + step])
                    L[layer, i + step] = self._g_soft(
                        L[layer + 1, i],
                        L[layer + 1, i + step],
                        u_llr[i],
                    )

            for i in range(N):
                if self.frozen_bits[i]:
                    u_llr[i] = self._large
                else:
                    u_llr[i] = L[0, i]

            u_hat = (u_llr < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            if self._check_early_stop(u_hat, llr_ch):
                num_iters = it
                break

        u_hat = (u_llr < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters

    def _check_early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, hard_ch)
