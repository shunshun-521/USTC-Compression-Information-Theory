"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode
from polar_common import bp_update_left, bp_update_right, generate_matrix


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.information_pos = list(np.where(self.frozen_bits == 0)[0])
        self.frozen_idx = list(np.where(self.frozen_bits == 1)[0])
        self.max_iter = max_iter
        self.alpha = alpha
        self.G = generate_matrix(self.n)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        alpha = self.alpha

        left_matrix = np.zeros((N, n + 1))
        right_matrix = np.zeros((N, n + 1))
        left_matrix[:, n] = llr_ch
        temp = [
            np.inf if i not in self.information_pos else 0.0 for i in range(N)
        ]
        right_matrix[:, 0] = temp

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for i in range(n):
                left_matrix[:, n - i - 1] = bp_update_left(
                    left_matrix[:, n - i],
                    right_matrix[:, n - i - 1],
                    n - i,
                    alpha,
                )
            for i in range(n):
                right_matrix[:, i + 1] = bp_update_right(
                    left_matrix[:, i + 1],
                    right_matrix[:, i],
                    i + 1,
                    alpha,
                )

            u_d_llr = left_matrix[:, 0] + right_matrix[:, 0]
            u_hat = (u_d_llr < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            num_iters = it
            if np.array_equal(x_hat, x_hard):
                break

        u_d_llr = left_matrix[:, 0] + right_matrix[:, 0]
        u_hat = (u_d_llr < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
