"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode, bit_reversal_permutation


class BPDecoder:
    """BP 译码器（信道 LLR 在内部做比特倒序以匹配编码器）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_mask = frozen_bits.astype(bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = llr_ch[self.br]
        n = self.n
        N = self.N
        LARGE = 1e6

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_mask, 0] = LARGE

        hard_ch = (llr_ch < 0).astype(int)
        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, step << 1):
                    for t in range(step):
                        a = i + t
                        b = a + step
                        L[a, j - 1] = self._f_ms(
                            R[a, j] + L[b, j], L[a, j]
                        )
                        L[b, j - 1] = self._f_ms(R[a, j], L[a, j]) + L[b, j]

            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, step << 1):
                    for t in range(step):
                        a = i + t
                        b = a + step
                        R[a, j + 1] = self._f_ms(
                            R[b, j] + L[b, j + 1], R[a, j]
                        )
                        R[b, j + 1] = self._f_ms(R[a, j], L[a, j + 1]) + R[b, j]

            u_hat = np.zeros(N, dtype=int)
            total = L[:, 0] + R[:, 0]
            u_hat[total < 0] = 1
            u_hat[self.frozen_mask] = 0

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        u_hat = np.zeros(N, dtype=int)
        total = L[:, 0] + R[:, 0]
        u_hat[total < 0] = 1
        u_hat[self.frozen_mask] = 0
        return u_hat, num_iters
