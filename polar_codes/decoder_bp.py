"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from decoder_sc import apply_llr_deperm, f_operation
from encoder import polar_encode, bit_reversal_permutation


def _minsum_f(a, b, alpha):
    return alpha * f_operation(a, b)


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]

    def decode(self, llr_ch):
        llr_raw = np.asarray(llr_ch, dtype=np.float64)
        llr = apply_llr_deperm(llr_raw)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))
        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)
        br = bit_reversal_permutation(N)

        for it in range(1, self.max_iter + 1):
            # L 消息：stage n-1 .. 0
            for stage in range(n - 1, -1, -1):
                step = 1 << stage
                for i in range(0, N, 2 * step):
                    for j in range(step):
                        a = i + j
                        b = a + step
                        L[a, stage] = _minsum_f(
                            R[a, stage + 1] + L[b, stage + 1],
                            L[a, stage + 1],
                            self.alpha,
                        )
                        L[b, stage] = (
                            _minsum_f(R[a, stage + 1], L[a, stage + 1], self.alpha)
                            + L[b, stage + 1]
                        )

            # R 消息：stage 0 .. n-1
            for stage in range(n):
                step = 1 << stage
                for i in range(0, N, 2 * step):
                    for j in range(step):
                        a = i + j
                        b = a + step
                        R[a, stage + 1] = _minsum_f(
                            R[b, stage + 1] + L[b, stage + 1],
                            R[a, stage],
                            self.alpha,
                        )
                        R[b, stage + 1] = (
                            _minsum_f(R[a, stage], L[a, stage + 1], self.alpha)
                            + R[b, stage]
                        )

            num_iters = it
            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard = (llr_raw < 0).astype(int)
            if np.array_equal(x_hat, hard):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
