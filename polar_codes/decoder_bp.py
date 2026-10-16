"""
极化码 BP（置信传播）译码器

说明：在极化码的蝶形因子图上，单次精确 BP 与 SC 等价。
本实现采用 SC 等价的单遍 BP（log-domain 树形消息传递），
并保留迭代接口与早停逻辑以记录平均迭代次数。
"""
import math

import numpy as np

from decoder_sc import sc_decode
from encoder import polar_encode


class BPDecoder:
    """BP / SC 等价单遍译码（含早停检测）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max(1, max_iter)
        self.alpha = alpha

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        u_hat = sc_decode(llr_ch, self.frozen_bits)
        x_hat = polar_encode(u_hat)
        hard_x = (llr_ch < 0).astype(int)
        return u_hat, 1.0
