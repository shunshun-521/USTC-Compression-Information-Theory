"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode, bit_reversal_permutation


class BPDecoder:
    """BP 译码器（分层 min-sum，含早停）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self._large = 1e6

    def _ms_f(self, a, b):
        return self.alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

    def decode(self, llr_ch):
        """
        返回 u_hat 与实际迭代次数。
        """
        llr_orig = np.asarray(llr_ch, dtype=np.float64)
        llr_in = llr_orig[bit_reversal_permutation(self.N)]
        hard_tx = (llr_orig < 0).astype(int)
        n, N = self.n, self.N

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_in
        R[0, :] = 0.0
        R[0, self.frozen_idx] = self._large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for s in range(n, 0, -1):
                step = 2 ** (s - 1)
                for i in range(0, N, 2 * step):
                    for j in range(step):
                        i1, i2 = i + j, i + j + step
                        L[s - 1, i1] = self._ms_f(
                            L[s, i1] + R[s - 1, i1], L[s, i2]
                        )
                        L[s - 1, i2] = self._ms_f(
                            R[s - 1, i1], L[s, i1]
                        ) + L[s, i2]

            for s in range(1, n + 1):
                step = 2 ** (s - 1)
                for i in range(0, N, 2 * step):
                    for j in range(step):
                        i1, i2 = i + j, i + j + step
                        R[s, i1] = self._ms_f(
                            R[s - 1, i2] + L[s, i2], R[s - 1, i1]
                        )
                        R[s, i2] = self._ms_f(
                            R[s - 1, i1], L[s, i1]
                        ) + R[s - 1, i2]

            u_hat = self._hard_decision(L[0, :] + R[0, :])
            if np.array_equal(polar_encode(u_hat), hard_tx):
                num_iters = it
                break

        u_hat = self._hard_decision(L[0, :] + R[0, :])
        return u_hat, num_iters

    def _hard_decision(self, llr):
        u_hat = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if llr[i] >= 0 else 1
        return u_hat
