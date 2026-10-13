"""
极化码 BP（置信传播）译码器
基于校验矩阵 H 的 min-sum BP，含早停（码字满足 Hx=0）
"""
import math
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from parity import get_parity_matrix, minsum_bp_decode
from decoder_sc import _butterfly_bits


class BPDecoder:
    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        assert 2 ** self.n == N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.info_indices = tuple(int(i) for i in np.where(~self.frozen_bits)[0])
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        H = get_parity_matrix(self.N, self.info_indices)
        x_hat = minsum_bp_decode(H, llr_ch, max_iter=self.max_iter)
        rev = bit_reversal_permutation(self.N)
        u_hat = _butterfly_bits(x_hat[rev])
        u_hat[self.frozen_bits] = 0

        hard_x = (llr_ch < 0).astype(int)
        num_iters = self.max_iter
        if np.array_equal(x_hat, hard_x):
            num_iters = 1
        return u_hat, num_iters
