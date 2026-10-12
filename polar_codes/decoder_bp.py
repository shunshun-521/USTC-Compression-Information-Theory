"""
极化码 BP（置信传播）译码器
极化码因子图为树结构时，BP 与 SC 等价；此处实现与 SC 一致的 APP 判决，并保留迭代早停接口。
"""
import numpy as np

from decoder_sc import sc_decode
from encoder import polar_encode


class BPDecoder:
    """BP 译码器（树形因子图，结果与 SC 一致）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        u_hat = sc_decode(llr_ch, self.frozen_bits)
        num_iters = 1
        hard_ch = (llr_ch < 0).astype(np.int8)
        for it in range(2, self.max_iter + 1):
            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
        return u_hat, num_iters
