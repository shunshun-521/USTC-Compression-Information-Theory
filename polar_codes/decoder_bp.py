"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from channel import hard_decision_llr
from encoder import bit_reversal_permutation, polar_encode


def min_sum_f(a, b, alpha=0.9375):
    """min-sum 近似的 f 运算"""
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 层（层 0 到层 n），每层 N 个节点。
    层 0：信源比特端；层 n：信道接收端。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]
        self.large = 1e6
        self.rev = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        N = self.N
        n = self.n
        alpha = self.alpha

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)

        L[n, :] = np.asarray(llr_ch, dtype=np.float64)[self.rev]
        R[0, :] = 0.0
        R[0, self.frozen_idx] = self.large

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for layer in range(n - 1, -1, -1):
                stride = 1 << layer
                for block in range(0, N, 2 * stride):
                    left = slice(block, block + stride)
                    right = slice(block + stride, block + 2 * stride)
                    L[layer, left] = min_sum_f(
                        R[layer + 1, left] + L[layer + 1, right],
                        L[layer + 1, left],
                        alpha,
                    )
                    L[layer, right] = (
                        min_sum_f(R[layer + 1, left], L[layer + 1, left], alpha)
                        + L[layer + 1, right]
                    )

            for layer in range(n):
                stride = 1 << layer
                for block in range(0, N, 2 * stride):
                    left = slice(block, block + stride)
                    right = slice(block + stride, block + 2 * stride)
                    R[layer + 1, left] = min_sum_f(
                        R[layer + 1, right] + L[layer + 1, right],
                        R[layer, left],
                        alpha,
                    )
                    R[layer + 1, right] = (
                        min_sum_f(R[layer, left], L[layer + 1, left], alpha)
                        + R[layer + 1, right]
                    )

            total_llr = L[0, :] + R[0, :]
            u_hat = np.zeros(N, dtype=int)
            u_hat[self.info_idx] = (total_llr[self.info_idx] < 0).astype(int)

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

        total_llr = L[0, :] + R[0, :]
        u_hat = np.zeros(N, dtype=int)
        u_hat[self.info_idx] = (total_llr[self.info_idx] < 0).astype(int)
        return u_hat, num_iters
