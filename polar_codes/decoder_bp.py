"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _minsum_f(x, y, alpha):
    sx = np.where(x >= 0, 1.0, -1.0)
    sy = np.where(y >= 0, 1.0, -1.0)
    return alpha * sx * sy * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def _run_iterations(self, llr_ch, fixed_prior=None, num_iter=None):
        """执行 BP 迭代，返回 (L, R, u_hat, iters)"""
        from channel import channel_llr_to_decoder

        llr_raw = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = channel_llr_to_decoder(llr_raw)
        n = self.n
        N = self.N
        if num_iter is None:
            num_iter = self.max_iter

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        if fixed_prior:
            for idx, bit in fixed_prior.items():
                R[idx, 0] = self.LARGE if bit == 0 else -self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, num_iter + 1):
            num_iters = it
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        idx = i + t
                        idx2 = idx + s
                        L[idx, j - 1] = _minsum_f(
                            R[idx, j] + L[idx2, j], L[idx, j], self.alpha
                        )
                        L[idx2, j - 1] = _minsum_f(R[idx, j], L[idx, j], self.alpha) + L[
                            idx2, j
                        ]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        idx = i + t
                        idx2 = idx + s
                        R[idx, j + 1] = _minsum_f(
                            R[idx2, j] + L[idx2, j + 1], R[idx, j], self.alpha
                        )
                        R[idx2, j + 1] = _minsum_f(R[idx, j], L[idx, j + 1], self.alpha) + R[
                            idx2, j
                        ]

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            if num_iter > 1:
                x_hat = polar_encode(u_hat)
                hard_ch = (llr_raw < 0).astype(int)
                if np.array_equal(x_hat, hard_ch):
                    break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return L, R, u_hat, num_iters, llr_ch

    def decode(self, llr_ch):
        _, _, u_hat, num_iters, _ = self._run_iterations(llr_ch)
        return u_hat, num_iters

    def bit_marginal(self, llr_ch, fixed_prior, phi):
        """在固定先验比特下，返回第 phi 位的后验 LLR"""
        L, R, _, _, _ = self._run_iterations(llr_ch, fixed_prior=fixed_prior, num_iter=1)
        return L[phi, 0] + R[phi, 0]
