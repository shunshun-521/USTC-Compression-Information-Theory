"""
极化码 BP（置信传播）译码器
基于因子图的迭代软抵消（SCAN）实现，含 min-sum 近似与早停
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import (
    _bit_reversed,
    _update_bits,
    _update_llrs,
)


class BPDecoder:
    """BP / SCAN 迭代译码器。"""

    def __init__(self, N, frozen_bits, max_iter=20, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        br = np.array([_bit_reversed(i, n) for i in range(N)])
        llr_work = llr_ch[br].copy()

        soft_llr = np.zeros(N, dtype=np.float64)
        decode_order = [_bit_reversed(i, n) for i in range(N)]
        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            L = np.zeros((N, n + 1), dtype=np.float64)
            B = np.zeros((N, n + 1), dtype=np.int8)
            L[:, 0] = llr_work

            for l in decode_order:
                _update_llrs(L, B, l, n)
                if self.frozen_bits[l]:
                    soft_llr[l] = self.alpha * soft_llr[l] + (1 - self.alpha) * 50.0
                else:
                    soft_llr[l] = self.alpha * soft_llr[l] + (1 - self.alpha) * L[l, n]

                B[l, n] = 0 if soft_llr[l] >= 0 else 1
                _update_bits(B, l, n)

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if soft_llr[i] >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break

        return u_hat.astype(int), num_iters
