"""
极化码 BP（置信传播）译码器：min-sum 近似 + 早停
"""
import numpy as np
import _shipeng_function as fn
from encoder import polar_encode


class BPDecoder:
    """BP 译码器（基于因子图消息传递）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.info_indices = list(np.where(self.frozen_bits == 0)[0])
        self.max_iter = int(max_iter)
        self.alpha = float(alpha)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        left = np.zeros((N, n + 1), dtype=np.float64)
        right = np.zeros((N, n + 1), dtype=np.float64)
        left[:, n] = llr_ch
        frozen_val = 0
        temp = np.array(
            [(1 - 2 * frozen_val) * np.inf if i not in self.info_indices else 0.0 for i in range(N)],
            dtype=np.float64,
        )
        right[:, 0] = temp

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for i in range(n):
                left[:, n - i - 1] = fn.bp_update_left(left[:, n - i], right[:, n - i - 1], n - i)
            for i in range(n):
                right[:, i + 1] = fn.bp_update_right(left[:, i + 1], right[:, i], i + 1)

            post = left[:, 0] + right[:, 0]
            u_hat = (post < 0).astype(int)
            for idx in range(N):
                if self.frozen_bits[idx]:
                    u_hat[idx] = 0

            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            num_iters = it
            if np.array_equal(x_hat, x_hard):
                break

        return u_hat, num_iters
