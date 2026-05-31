"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from decoder_sc import f_operation, _prepare_llr
from encoder import polar_encode


LARGE = 1e6


class BPDecoder:
    """BP 译码器（min-sum 近似 + 早停）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_indices = np.where(self.frozen_bits)[0]
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _f_min_sum(self, x, y):
        """min-sum f 运算（带 alpha 修正）"""
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        """主译码函数"""
        N, n = self.N, self.n
        llr_ch = _prepare_llr(llr_ch, self.N)

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_indices, 0] = LARGE

        num_iters = 0
        for it in range(1, self.max_iter + 1):
            num_iters = it

            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        idx_s = i + k + s
                        L[idx, j - 1] = self._f_min_sum(
                            R[idx, j] + L[idx_s, j], L[idx, j]
                        )
                        L[idx_s, j - 1] = self._f_min_sum(
                            R[idx, j], L[idx, j]
                        ) + L[idx_s, j]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        idx_s = i + k + s
                        R[idx, j + 1] = self._f_min_sum(
                            R[idx_s, j] + L[idx_s, j + 1], R[idx, j]
                        )
                        R[idx_s, j + 1] = self._f_min_sum(
                            R[idx, j], L[idx, j + 1]
                        ) + R[idx_s, j]

            u_hat = self._hard_decision(L, R)
            if self._early_stop(u_hat, llr_ch):
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters

    def _hard_decision(self, L, R):
        """最左列硬判决"""
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_indices] = 0
        return u_hat

    def _early_stop(self, u_hat, llr_ch):
        """重新编码后与信道硬判决一致则早停"""
        x_hat = polar_encode(u_hat)
        llr_prep = _prepare_llr(llr_ch, self.N)
        hard_ch = (llr_prep < 0).astype(int)
        return np.array_equal(x_hat, hard_ch)
