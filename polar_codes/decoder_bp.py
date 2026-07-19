"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from decoder_sc import (
    f_operation_min_sum,
    _prepare_channel_llr,
    _bit_reversed_index,
    _active_llr_level,
    _active_bit_level,
)
from encoder import polar_encode
from channel import hard_decision_llr


class BPDecoder:
    """
    BP 译码器（迭代软 SC 因子图传播 + min-sum 近似）。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _f(self, a, b):
        return self.alpha * f_operation_min_sum(a, b)

    def _g(self, a, b, u):
        return (1.0 - 2.0 * u) * a + b

    def _update_llrs_full(self, L, B, n):
        """对所有节点执行一次 LLR 自顶向下更新（使用当前 B）。"""
        N = L.shape[0]
        for l in range(N):
            for s in range(n - _active_llr_level(l, n), n):
                block_size = 2 ** (s + 1)
                branch_size = block_size // 2
                for j in range(l, N, block_size):
                    if j % block_size < branch_size:
                        L[j, s + 1] = self._f(L[j, s], L[j + branch_size, s])
                    else:
                        L[j, s + 1] = self._g(
                            L[j - branch_size, s],
                            L[j, s],
                            B[j - branch_size, s + 1],
                        )

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        channel = _prepare_channel_llr(llr_ch, N)

        L = np.full((N, n + 1), np.nan, dtype=np.float64)
        B = np.zeros((N, n + 1), dtype=np.int32)
        L[:, 0] = channel

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            self._update_llrs_full(L, B, n)

            for i in range(N):
                l = _bit_reversed_index(i, n)
                if self.frozen_bits[l]:
                    B[l, n] = 0
                else:
                    llr_bit = L[l, n]
                    if np.isnan(llr_bit):
                        llr_bit = 0.0
                    B[l, n] = 0 if llr_bit >= 0 else 1

                if l >= N / 2:
                    for s in range(n, n - _active_bit_level(l, n), -1):
                        block_size = 2 ** s
                        branch_size = block_size // 2
                        for j in range(l, -1, -block_size):
                            if j % block_size >= branch_size:
                                B[j - branch_size, s - 1] = (
                                    int(B[j, s]) ^ int(B[j - branch_size, s])
                                )
                                B[j, s - 1] = B[j, s]

            u_hat = B[:, n].astype(int)

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_decision_llr(llr_ch)):
                num_iters = it
                break

        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
