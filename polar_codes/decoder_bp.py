"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _ms_f(a, b, alpha):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    sign_a = np.sign(a)
    sign_b = np.sign(b)
    sign_a = np.where(sign_a == 0, 1.0, sign_a)
    sign_b = np.where(sign_b == 0, 1.0, sign_b)
    return alpha * sign_a * sign_b * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    @staticmethod
    def _hard_codeword(llr_ch):
        hard_x = np.zeros(len(llr_ch), dtype=int)
        br = bit_reversal_permutation(len(llr_ch))
        hard_x[br] = (llr_ch < 0).astype(int)
        return hard_x

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        alpha = self.alpha

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for stage in range(n - 1, -1, -1):
                block = 1 << (stage + 1)
                half = 1 << stage
                for base in range(0, N, block):
                    left = slice(base, base + half)
                    right = slice(base + half, base + block)
                    L[left, stage] = _ms_f(
                        R[left, stage] + L[right, stage + 1],
                        L[left, stage + 1],
                        alpha,
                    )
                    L[right, stage] = (
                        _ms_f(R[left, stage], L[left, stage + 1], alpha)
                        + L[right, stage + 1]
                    )

            for stage in range(0, n):
                block = 1 << (stage + 1)
                half = 1 << stage
                for base in range(0, N, block):
                    left = slice(base, base + half)
                    right = slice(base + half, base + block)
                    R[left, stage + 1] = _ms_f(
                        R[right, stage] + L[right, stage + 1],
                        R[left, stage],
                        alpha,
                    )
                    R[right, stage + 1] = (
                        _ms_f(R[left, stage], L[left, stage + 1], alpha)
                        + R[right, stage]
                    )

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            if np.array_equal(polar_encode(u_hat), self._hard_codeword(llr_ch)):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
