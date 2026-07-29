"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import f_operation
from encoder import polar_encode
from channel import hard_decision_llr


def _boxplus_f(a, b):
    """f 函数（对数域 box-plus）。"""
    from decoder_sc import _logdomain_sum
    return _logdomain_sum(a + b, 0.0) - _logdomain_sum(a, b)

class BPDecoder:
    """BP 译码器（因子图 flooding 调度）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e9

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64).copy()
        n = self.n
        N = self.N

        # stage s=0..n, s=0 为信源端, s=n 为信道端
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(self.max_iter):
            num_iters = it + 1
            L_new = L.copy()
            R_new = R.copy()

            # 右向左：更新 L (stage n -> 1)
            for s in range(n, 0, -1):
                half = 1 << (s - 1)
                for base in range(0, N, 2 * half):
                    for j in range(half):
                        i = base + j
                        L_new[i, s - 1] = self._f_ms(
                            R[i, s] + L[i + half, s], L[i, s]
                        )
                        L_new[i + half, s - 1] = self._f_ms(
                            R[i, s], L[i, s]
                        ) + L[i + half, s]

            L = L_new

            # 左向右：更新 R (stage 0 -> n-1)
            for s in range(0, n):
                half = 1 << s
                for base in range(0, N, 2 * half):
                    for j in range(half):
                        i = base + j
                        R_new[i, s + 1] = self._f_ms(
                            R[i + half, s] + L[i + half, s + 1], R[i, s]
                        )
                        R_new[i + half, s + 1] = self._f_ms(
                            R[i, s], L[i, s + 1]
                        ) + R[i + half, s]

            R = R_new

            total = L[:, 0] + R[:, 0]
            u_hat = np.where(total >= 0, 0, 1).astype(np.int8)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_decision_llr(llr_ch)):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.where(total >= 0, 0, 1).astype(np.int8)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
