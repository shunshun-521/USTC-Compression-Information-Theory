"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _ms_f(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _hard_bits_from_llr(self, llr_ch):
        return (llr_ch < 0).astype(int)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = llr_ch[bit_reversal_permutation(self.N)]

        n = self.n
        N = self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for stage in range(n, 0, -1):
                block = 1 << stage
                half = block >> 1
                for start in range(0, N, block):
                    for k in range(half):
                        i = start + k
                        j = i + half
                        L[i, stage - 1] = _ms_f(
                            L[i, stage], L[j, stage] + R[j, stage], self.alpha
                        )
                        L[j, stage - 1] = _ms_f(R[i, stage], L[i, stage], self.alpha) + L[j, stage]

            for stage in range(0, n):
                block = 1 << (stage + 1)
                half = block >> 1
                for start in range(0, N, block):
                    for k in range(half):
                        i = start + k
                        j = i + half
                        R[i, stage + 1] = _ms_f(
                            R[i, stage], L[j, stage + 1] + R[j, stage], self.alpha
                        )
                        R[j, stage + 1] = _ms_f(R[i, stage], L[i, stage + 1], self.alpha) + R[j, stage]

            for i in range(N):
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
            u_hat[self.frozen_idx] = 0

            if np.array_equal(polar_encode(u_hat), self._hard_bits_from_llr(llr_ch)):
                num_iters = it
                break

        for i in range(N):
            u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
        u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
