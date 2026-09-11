"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode


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
        self._large = 1e10

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：
            u_hat: 长度 N 的估计源序列
            num_iters: 实际迭代次数
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        m = self.n

        L = np.zeros((N, m + 1), dtype=np.float64)
        R = np.zeros((N, m + 1), dtype=np.float64)
        L[:, m] = llr_ch
        R[self.frozen_bits, 0] = self._large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for s in range(m - 1, -1, -1):
                step = 2 ** s
                for i in range(0, N, 2 * step):
                    for j in range(step):
                        a = i + j
                        b = i + j + step
                        L[a, s] = self._f_min_sum(L[a, s + 1] + R[a, s], L[b, s + 1])
                        L[b, s] = self._f_min_sum(R[a, s], L[a, s + 1]) + L[b, s + 1]

            for s in range(m):
                step = 2 ** s
                for i in range(0, N, 2 * step):
                    for j in range(step):
                        a = i + j
                        b = i + j + step
                        R[a, s + 1] = self._f_min_sum(L[b, s + 1] + R[b, s], R[a, s])
                        R[b, s + 1] = self._f_min_sum(R[a, s], L[b, s + 1]) + R[b, s]

            total = L[:, 0] + R[:, 0]
            u_hat[~self.frozen_bits] = (total[~self.frozen_bits] < 0).astype(int)

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        return u_hat, num_iters
