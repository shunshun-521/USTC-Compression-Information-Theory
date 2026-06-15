"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode
from decoder_sc import _preprocess_channel_llr
from sc_core import bp_update_left, bp_update_right

LARGE = 1e6


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.information_pos = np.where(~self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha

    def _hard_decision(self, llr):
        return (llr < 0).astype(int)

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, num_iters"""
        llr_proc = _preprocess_channel_llr(llr_ch)
        n, N = self.n, self.N
        left_matrix = np.zeros((N, n + 1), dtype=np.float64)
        right_matrix = np.zeros((N, n + 1), dtype=np.float64)
        left_matrix[:, n] = llr_proc
        right_matrix[:, 0] = 0.0
        right_matrix[self.frozen_bits, 0] = LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for i in range(n):
                left_matrix[:, n - i - 1] = bp_update_left(
                    left_matrix[:, n - i],
                    right_matrix[:, n - i - 1],
                    n - i,
                )

            for i in range(n):
                right_matrix[:, i + 1] = bp_update_right(
                    left_matrix[:, i + 1],
                    right_matrix[:, i],
                    i + 1,
                )

            total = left_matrix[:, 0] + right_matrix[:, 0]
            u_hat = self._hard_decision(total)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = self._hard_decision(llr_ch)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = left_matrix[:, 0] + right_matrix[:, 0]
        u_hat = self._hard_decision(total)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
