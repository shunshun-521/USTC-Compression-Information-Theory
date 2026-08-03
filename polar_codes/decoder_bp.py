"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
import _ref_function as ref_function


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.br = bit_reversal_permutation(N)
        self.LARGE = 1e6

    def decode(self, llr_ch):
        """
        主译码函数。

        返回：
            u_hat: 长度 N 的估计源序列
            num_iters: 实际迭代次数
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        y_llr = llr_ch[self.br].copy()
        N = self.N
        n = self.n

        left_matrix = np.zeros((N, n + 1))
        right_matrix = np.zeros((N, n + 1))
        left_matrix[:, n] = y_llr
        temp = [self.LARGE if self.frozen_bits[i] else 0.0 for i in range(N)]
        right_matrix[:, 0] = temp

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for iteration in range(self.max_iter):
            for i in range(n):
                left_matrix[:, n - i - 1] = ref_function.bp_update_left(
                    left_matrix[:, n - i], right_matrix[:, n - i - 1], n - i)

            for i in range(n):
                right_matrix[:, i + 1] = ref_function.bp_update_right(
                    left_matrix[:, i + 1], right_matrix[:, i], i + 1)

            u_d_llr = left_matrix[:, 0] + right_matrix[:, 0]
            for i in range(N):
                u_hat[i] = 0 if self.frozen_bits[i] or u_d_llr[i] >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_decision = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_decision):
                num_iters = iteration + 1
                break
        else:
            u_d_llr = left_matrix[:, 0] + right_matrix[:, 0]
            for i in range(N):
                u_hat[i] = 0 if self.frozen_bits[i] or u_d_llr[i] >= 0 else 1

        return u_hat, num_iters
