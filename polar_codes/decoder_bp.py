"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器（参考 DOAN GLOCOM'18 因子图 PE 更新规则）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """主译码函数"""
        N, n = self.N, self.n
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        L = np.zeros((N, n), dtype=np.float64)
        R = np.zeros((N, n), dtype=np.float64)

        L[:, n - 1] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = 1e6

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for l in range(n - 2, -1, -1):
                step = 1 << l
                for i in range(0, N, 2 * step):
                    j = i + step
                    L[i, l] = self._f_min_sum(
                        L[i, l + 1],
                        R[j, l] + L[j, l + 1],
                    )
                    L[j, l] = self._f_min_sum(L[i, l + 1], R[i, l]) + L[j, l + 1]

            for l in range(0, n - 1):
                step = 1 << l
                for i in range(0, N, 2 * step):
                    j = i + step
                    R[i, l + 1] = self._f_min_sum(
                        R[i, l],
                        L[j, l + 1] + R[j, l],
                    )
                    R[j, l + 1] = self._f_min_sum(R[i, l], L[i, l + 1]) + R[j, l]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
