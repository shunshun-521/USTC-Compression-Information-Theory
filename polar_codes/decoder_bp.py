"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    _prepare_channel_llrs,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
)
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
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.LARGE = 1e6

    def _f_min_sum(self, x, y):
        return self.alpha * f_operation(x, y)

    def _update_partial_bits(self, B, u_hat):
        """SC 风格部分和回传，将冻结位信息注入因子图。"""
        N, n = self.N, self.n
        for i in range(N):
            l = _bit_reversed(i, n)
            B[l, n] = u_hat[l]
            if l < N // 2:
                continue
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                        B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_natural = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = _prepare_channel_llrs(llr_natural)
        N = self.N
        n = self.n

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        B = np.zeros((N, n + 1), dtype=int)

        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_idx] = self.LARGE

        u_hat = np.zeros(N, dtype=int)
        num_iters = self.max_iter

        for iteration in range(1, self.max_iter + 1):
            R[0, self.frozen_idx] = self.LARGE

            for lamb in range(n - 1, -1, -1):
                block = 1 << lamb
                for phi in range(0, N, 2 * block):
                    for omega in range(block):
                        a = phi + omega
                        b = a + block
                        if (phi // (2 * block)) % 2 == 0:
                            L[lamb, a] = self._f_min_sum(
                                R[lamb + 1, a] + L[lamb + 1, b],
                                L[lamb + 1, a],
                            )
                            L[lamb, b] = (
                                self._f_min_sum(R[lamb + 1, a], L[lamb + 1, a])
                                + L[lamb + 1, b]
                            )
                        else:
                            u_bit = B[a, lamb]
                            L[lamb, a] = self._f_min_sum(
                                R[lamb + 1, a] + L[lamb + 1, b],
                                L[lamb + 1, a],
                            )
                            L[lamb, b] = g_operation(
                                self._f_min_sum(R[lamb + 1, a], L[lamb + 1, a]),
                                L[lamb + 1, b],
                                u_bit,
                            )

            for lamb in range(0, n):
                block = 1 << lamb
                for phi in range(0, N, 2 * block):
                    for omega in range(block):
                        a = phi + omega
                        b = a + block
                        R[lamb + 1, a] = self._f_min_sum(
                            R[lamb, b] + L[lamb + 1, b],
                            R[lamb, a],
                        )
                        R[lamb + 1, b] = (
                            self._f_min_sum(R[lamb, a], L[lamb + 1, a])
                            + R[lamb, b]
                        )

            total = L[0, :] + R[0, :]
            u_hat[:] = 0
            u_hat[total < 0] = 1
            u_hat[self.frozen_idx] = 0
            self._update_partial_bits(B, u_hat)

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_natural < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = iteration
                break
        else:
            total = L[0, :] + R[0, :]
            u_hat[:] = 0
            u_hat[total < 0] = 1
            u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
