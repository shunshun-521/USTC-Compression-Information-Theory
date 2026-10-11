"""
极化码 BP（置信传播）译码器
"""
import numpy as np

from bp_backend import bp_decoder


class BPDecoder:
    """BP 译码器（因子图 min-sum，含早停）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits)
        self.max_iter = max_iter
        self.alpha = alpha
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        info_pos = self.info_indices.tolist()
        frozen_val = 0
        decode_para = [self.max_iter, "g_matrix"]
        u_hat = bp_decoder(llr_ch, info_pos, frozen_val, decode_para, 0)
        u_hat = np.asarray(u_hat, dtype=int)
        return u_hat, None
