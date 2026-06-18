"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import _prepare_llr
from encoder import polar_encode


class BPDecoder:
    """BP 译码器（基于参考实现的因子图更新）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        import _ref_function as function

        N = self.N
        n = self.n
        y_llr = _prepare_llr(llr_ch)

        left_matrix = np.zeros((N, n + 1), dtype=np.float64)
        right_matrix = np.zeros((N, n + 1), dtype=np.float64)
        left_matrix[:, n] = y_llr

        temp = np.zeros(N, dtype=np.float64)
        temp[self.frozen_bits] = np.inf
        right_matrix[:, 0] = temp

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            for i in range(n):
                left_matrix[:, n - i - 1] = function.bp_update_left(
                    left_matrix[:, n - i],
                    right_matrix[:, n - i - 1],
                    n - i,
                )
            for i in range(n):
                right_matrix[:, i + 1] = function.bp_update_right(
                    left_matrix[:, i + 1],
                    right_matrix[:, i],
                    i + 1,
                )

            u_d_llr = left_matrix[:, 0] + right_matrix[:, 0]
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if u_d_llr[i] >= 0 else 1

            x_d_llr = left_matrix[:, n] + right_matrix[:, n]
            x_d = (x_d_llr < 0).astype(int)
            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, x_d):
                break

        u_d_llr = left_matrix[:, 0] + right_matrix[:, 0]
        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if u_d_llr[i] >= 0 else 1

        return u_hat.astype(int), num_iters
