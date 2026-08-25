"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import bit_reversal_permutation, polar_encode


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6

    def _f_min_sum(self, a, b):
        sa = np.where(a >= 0, 1.0, -1.0)
        sb = np.where(b >= 0, 1.0, -1.0)
        return self.alpha * sa * sb * np.minimum(np.abs(a), np.abs(b))

    def _update_l_messages(self, L, R):
        n = self.n
        N = self.N
        for j in range(n - 1, -1, -1):
            step = 1 << j
            for i in range(0, N, 2 * step):
                for k in range(step):
                    idx_u = i + k
                    idx_v = i + k + step
                    L[idx_u, j] = self._f_min_sum(
                        R[idx_u, j] + L[idx_v, j + 1], L[idx_u, j + 1]
                    )
                    L[idx_v, j] = self._f_min_sum(
                        R[idx_u, j], L[idx_u, j + 1]
                    ) + L[idx_v, j + 1]

    def _update_r_messages(self, L, R):
        n = self.n
        N = self.N
        for j in range(1, n):
            step = 1 << (j - 1)
            for i in range(0, N, 2 * step):
                for k in range(step):
                    idx_u = i + k
                    idx_v = i + k + step
                    R[idx_u, j - 1] = self._f_min_sum(
                        R[idx_v, j] + L[idx_v, j + 1], R[idx_u, j - 1]
                    )
                    R[idx_v, j - 1] = self._f_min_sum(
                        R[idx_u, j - 1], L[idx_u, j + 1]
                    ) + R[idx_v, j]

    def _hard_decision(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits == 1] = 0
        return u_hat

    def _check_early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, hard_ch)

    def decode(self, llr_ch):
        """返回：(u_hat, num_iters)"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        # 与 SC 译码器一致：将信道 LLR 映射到自然序
        br = bit_reversal_permutation(N)
        llr_nat = np.zeros(N, dtype=np.float64)
        for i in range(N):
            llr_nat[br[i]] = llr_ch[i]

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_nat
        R[:, 0] = 0.0
        R[self.frozen_bits == 1, 0] = self._large

        num_iters = 0
        for it in range(self.max_iter):
            num_iters = it + 1
            self._update_l_messages(L, R)
            self._update_r_messages(L, R)
            u_hat = self._hard_decision(L, R)
            if self._check_early_stop(u_hat, llr_ch):
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters
