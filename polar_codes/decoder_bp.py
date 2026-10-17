"""
极化码 BP（置信传播）译码器，min-sum 近似 + 早停
"""
import numpy as np
from _codec_core import bp_decoder as _bp_decoder


def _info_indices(frozen_bits):
    return np.where(np.asarray(frozen_bits) == 0)[0]


class BPDecoder:
    """BP 译码器（g_matrix 早停：重编码与信道硬判决一致）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.info_idx = _info_indices(self.frozen_bits)
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        decode_para = [self.max_iter, "g_matrix"]
        out = _bp_decoder(llr_ch, self.info_idx, 0, decode_para, 0)
        if isinstance(out, tuple):
            u_hat, num_iters = out
        else:
            u_hat, num_iters = out, self.max_iter
        u_hat = np.asarray(u_hat, dtype=int)
        for i in range(self.N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
        return u_hat, int(num_iters)
