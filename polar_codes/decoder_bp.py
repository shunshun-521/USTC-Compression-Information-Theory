"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import f_operation


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    列 0：信源比特端；列 n：信道接收端。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self.large = 1e6

    def _f_ms(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        """
        主译码函数。

        返回：
            u_hat: 长度 N 的估计源序列
            num_iters: 实际迭代次数
        """
        N, n = self.N, self.n
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_dec = llr_ch[self.br]

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_dec
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, step * 2):
                    for k in range(step):
                        idx = i + k
                        partner = idx + step
                        L[idx, j - 1] = self._f_ms(
                            R[idx, j] + L[partner, j],
                            L[idx, j],
                        )
                        L[partner, j - 1] = self._f_ms(
                            R[idx, j],
                            L[idx, j],
                        ) + L[partner, j]

            for j in range(1, n + 1):
                step = 1 << (j - 1)
                for i in range(0, N, step * 2):
                    for k in range(step):
                        idx = i + k
                        partner = idx + step
                        R[idx, j] = self._f_ms(
                            R[partner, j] + L[partner, j],
                            R[idx, j - 1],
                        )
                        R[partner, j] = self._f_ms(
                            R[idx, j - 1],
                            L[idx, j],
                        ) + R[partner, j]

            u_hat = self._hard_decision(L, R)
            if self._early_stop(u_hat, llr_dec):
                num_iters = it
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters

    def _hard_decision(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat

    def _early_stop(self, u_hat, llr_dec):
        x_hat = polar_encode(u_hat)
        br = self.br
        hard = (llr_dec < 0).astype(int)
        x_from_llr = np.zeros(self.N, dtype=int)
        x_from_llr[br] = hard
        return np.array_equal(x_hat, x_from_llr)
