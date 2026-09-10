"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode
from decoder_sc import f_operation, _bit_reversed, _active_llr_level, _active_bit_level


class BPDecoder:
    """
    BP 译码器（基于 Permuted SCD 因子图结构）。
    因子图有 n+1 列，每列 N 个节点。
    """

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def _f_ms(self, a, b):
        return self.alpha * f_operation(a, b)

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, num_iters"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, 0] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            # 右到左更新 L
            for s in range(n):
                block_size = 1 << (s + 1)
                branch_size = block_size // 2
                for j in range(0, N, block_size):
                    for k in range(branch_size):
                        idx_top = j + k
                        idx_btm = j + k + branch_size
                        L[idx_top, s + 1] = self._f_ms(
                            R[idx_top, s] + L[idx_btm, s], L[idx_top, s]
                        )
                        L[idx_btm, s + 1] = (
                            self._f_ms(R[idx_top, s], L[idx_top, s]) + L[idx_btm, s]
                        )

            # 左到右更新 R
            for s in range(n - 1, -1, -1):
                block_size = 1 << (s + 1)
                branch_size = block_size // 2
                for j in range(0, N, block_size):
                    for k in range(branch_size):
                        idx_top = j + k
                        idx_btm = j + k + branch_size
                        R[idx_top, s + 1] = self._f_ms(
                            R[idx_btm, s] + L[idx_btm, s + 1], R[idx_top, s]
                        )
                        R[idx_btm, s + 1] = (
                            self._f_ms(R[idx_top, s], L[idx_top, s + 1]) + R[idx_btm, s]
                        )

            # 早停
            total = L[:, n] + R[:, n]
            for phi in range(N):
                if self.frozen_bits[phi]:
                    u_hat[phi] = 0
                else:
                    l = _bit_reversed(phi, n)
                    u_hat[phi] = 0 if total[l] >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        total = L[:, n] + R[:, n]
        for phi in range(N):
            if self.frozen_bits[phi]:
                u_hat[phi] = 0
            else:
                l = _bit_reversed(phi, n)
                u_hat[phi] = 0 if total[l] >= 0 else 1

        return u_hat, num_iters
