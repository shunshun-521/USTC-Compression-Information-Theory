"""
极化码 BP（置信传播）译码器
基于 SC 软反馈迭代（min-sum 风格），含早停机制
"""
import math
import numpy as np

from encoder import polar_encode
from decoder_sc import sc_decode


class BPDecoder:
    """BP 译码器：SC 初始估计 + 码字一致性反馈迭代"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        L = llr_ch.copy()
        num_iters = self.max_iter
        u_hat = np.zeros(self.N, dtype=int)

        for it in range(1, self.max_iter + 1):
            u_hat = sc_decode(L, self.frozen_bits)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)

            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

            mismatch = x_hat != hard_ch
            if not np.any(mismatch):
                num_iters = it
                break

            for i in range(self.N):
                if mismatch[i]:
                    if hard_ch[i] == 0:
                        L[i] += self.alpha * abs(llr_ch[i] + 1.0)
                    else:
                        L[i] -= self.alpha * abs(llr_ch[i] - 1.0)
            num_iters = it

        return u_hat, num_iters
