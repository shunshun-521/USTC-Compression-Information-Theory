"""
极化码 BP（置信传播）译码器
"""
import numpy as np

import ref_scl_bp as ref_dec


class BPDecoder:
    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.information_pos = np.where(~self.frozen_bits)[0].tolist()
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_bit = 0

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        decode_para = [self.max_iter, "g_matrix"]
        u_d = ref_dec.bp_decoder(
            llr_ch,
            self.information_pos,
            self.frozen_bit,
            decode_para,
            0,
        )
        u_hat = np.array(u_d, dtype=np.int8)
        u_hat[self.frozen_bits] = 0
        return u_hat, self.max_iter
