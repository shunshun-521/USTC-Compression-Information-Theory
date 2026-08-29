"""
极化码 BP（置信传播）译码器
基于因子图 stage 更新（IEEE 公式 3-6），min-sum 近似，含早停
"""
import numpy as np
from encoder import polar_encode
from channel import hard_decision_llr
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e7

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def _h(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """返回 (u_hat, num_iters)"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        # L[:, s]: 右到左消息，s=n 为信道端；R[:, s]: 左到右消息，s=0 为信源端
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # 右到左更新 L（s = n-1 .. 0）
            for s in range(n - 1, -1, -1):
                stride = 1 << s
                for block in range(0, N, 2 * stride):
                    i = block
                    j = block + stride
                    L[i, s] = self._h(L[i, s + 1], L[j, s + 1] + R[j, s])
                    L[j, s] = self._h(R[i, s], L[i, s + 1] + L[j, s + 1])

            # 左到右更新 R（s = 0 .. n-1）
            for s in range(0, n):
                stride = 1 << s
                for block in range(0, N, 2 * stride):
                    i = block
                    j = block + stride
                    R[i, s + 1] = self._h(R[i, s], L[j, s + 1] + R[j, s])
                    R[j, s + 1] = self._h(R[i, s], L[i, s + 1] + R[j, s])

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            if np.array_equal(polar_encode(u_hat), hard_decision_llr(llr_ch)):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
