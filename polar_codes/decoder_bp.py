"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode, bit_reversal_permutation


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        if 2 ** self.n != N:
            raise ValueError("N must be a power of 2")
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    @staticmethod
    def _f_min(a, b, alpha):
        return alpha * np.sign(a) * np.sign(b) * min(abs(a), abs(b))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        llr = llr_ch[br]

        n = self.n
        N = self.N
        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))
        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, step << 1):
                    for t in range(step):
                        idx = i + t
                        s = idx + step
                        L[idx, j - 1] = self._f_min(
                            R[idx, j] + L[s, j], L[idx, j], self.alpha
                        )
                        L[s, j - 1] = self._f_min(R[idx, j], L[idx, j], self.alpha) + L[s, j]

            for j in range(1, n + 1):
                step = 1 << (j - 1)
                for i in range(0, N, step << 1):
                    for t in range(step):
                        idx = i + t
                        s = idx + step
                        R[idx, j] = self._f_min(
                            R[s, j] + L[s, j], R[idx, j - 1], self.alpha
                        )
                        R[s, j] = self._f_min(R[idx, j - 1], L[idx, j], self.alpha) + R[s, j]

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard):
                num_iters = it
                break

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            u_hat[i] = 0 if total >= 0 else 1
        u_hat[self.frozen_bits] = 0

        return u_hat, num_iters
