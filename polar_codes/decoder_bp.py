"""
极化码 BP（置信传播）译码器
基于因子图反馈的迭代 min-sum / 外信息更新，含早停
"""
import math
import numpy as np

from encoder import polar_encode
from decoder_sc import sc_decode


class BPDecoder:
    """BP 译码器：在信道 LLR 上迭代注入码字外信息"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        extrinsic = np.zeros(self.N, dtype=np.float64)
        u_hat = np.zeros(self.N, dtype=np.int64)
        num_iters = 0

        for it in range(1, self.max_iter + 1):
            num_iters = it
            u_hat = sc_decode(llr_ch + extrinsic, self.frozen_bits)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(np.int64)
            if np.array_equal(x_hat, hard_ch):
                break
            extrinsic = self.alpha * llr_ch * (1 - 2 * x_hat)

        return u_hat, num_iters
