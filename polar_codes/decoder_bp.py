"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _f_min_sum(x, y, alpha):
    """min-sum f 运算。"""
    return alpha * np.where(
        x * y >= 0,
        np.minimum(np.abs(x), np.abs(y)),
        -np.minimum(np.abs(x), np.abs(y)),
    )


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），列 0 为信源端，列 n 为信道端。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e7
        self.brp = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        """主译码函数。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_internal = llr_ch[self.brp]

        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_internal
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            # 左→右更新 R
            for j in range(n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        idx2 = idx + s
                        R[idx, j + 1] = _f_min_sum(
                            R[idx2, j] + L[idx2, j + 1],
                            R[idx, j],
                            self.alpha,
                        )
                        R[idx2, j + 1] = (
                            _f_min_sum(R[idx, j], L[idx, j + 1], self.alpha)
                            + R[idx2, j]
                        )

            # 右→左更新 L
            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        idx2 = idx + s
                        L[idx, j] = _f_min_sum(
                            R[idx, j] + L[idx2, j + 1],
                            L[idx, j + 1],
                            self.alpha,
                        )
                        L[idx2, j] = (
                            _f_min_sum(R[idx, j], L[idx, j + 1], self.alpha)
                            + L[idx2, j + 1]
                        )

            u_hat = self._hard_decision(L, R)
            if np.array_equal(polar_encode(u_hat), (llr_ch < 0).astype(int)):
                num_iters = it
                return u_hat, num_iters

        return self._hard_decision(L, R), num_iters

    def _hard_decision(self, L, R):
        u_hat = np.zeros(self.N, dtype=int)
        total = L[:, 0] + R[:, 0]
        for i in range(self.N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total[i] >= 0 else 1
        return u_hat
