"""
极化码 BP（置信传播）译码器
采用 LLR 反馈迭代（SCAN 风格近似），含早停
"""
import math
import numpy as np
from decoder_sc import sc_decode
from encoder import polar_encode


class BPDecoder:
    """BP 迭代译码器（SC + LLR 反馈）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        llr_natural = np.asarray(llr_ch, dtype=np.float64)
        feedback = np.zeros(self.N, dtype=np.float64)
        u_hat = np.zeros(self.N, dtype=np.int8)
        num_iters = 0

        for it in range(self.max_iter):
            num_iters = it + 1
            llr_mod = llr_natural + self.alpha * feedback
            u_hat = sc_decode(llr_mod, self.frozen_bits)

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_natural < 0).astype(np.int8)
            if np.array_equal(x_hat, hard_ch):
                break

            feedback = (1.0 - 2.0 * u_hat) * np.abs(llr_natural)

        return u_hat, num_iters
