"""
极化码 BP（置信传播）译码器
"""
import numpy as np

import _ref_decoder as _ref


class BPDecoder:
    """BP 译码器（基于参考实现的 min-sum BP）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.info_pos = np.where(~self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        decode_para = [self.max_iter, "g_matrix"]
        u_d, num_iters = _ref.bp_decoder(
            llr_ch,
            self.info_pos,
            0,
            decode_para,
            0,
        )
        u_hat = np.array([0 if u_d[i] == 0 else 1 for i in range(self.N)], dtype=np.int8)
        return u_hat, num_iters
