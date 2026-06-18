"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation


def f_min_sum(La, Lb, alpha=0.9375):
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return alpha * sa * sb * np.minimum(np.abs(La), np.abs(Lb))


class BPDecoder:
    """
    BP 译码器。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        self.frozen_indices = np.where(self.frozen_bits.astype(bool) if self.frozen_bits.dtype == bool
                                       else self.frozen_bits == 1)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def decode(self, llr_ch):
        N = self.N
        n = self.n
        alpha = self.alpha
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_indices, 0] = self.large

        def hard_bits():
            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_indices] = 0
            return u_hat

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for stage in range(n, 0, -1):
                step = 1 << (stage - 1)
                for i in range(0, N, step << 1):
                    for j in range(step):
                        idx = i + j
                        s = step
                        L[idx, stage - 1] = f_min_sum(
                            R[idx, stage - 1] + L[idx, stage],
                            L[idx + s, stage],
                            alpha,
                        )
                        L[idx + s, stage - 1] = (
                            f_min_sum(R[idx, stage - 1], L[idx, stage], alpha)
                            + L[idx + s, stage]
                        )

            for stage in range(0, n):
                step = 1 << stage
                for i in range(0, N, step << 1):
                    for j in range(step):
                        idx = i + j
                        s = step
                        R[idx, stage + 1] = f_min_sum(
                            R[idx + s, stage] + L[idx + s, stage + 1],
                            R[idx, stage],
                            alpha,
                        )
                        R[idx + s, stage + 1] = (
                            f_min_sum(R[idx, stage], L[idx, stage + 1], alpha)
                            + R[idx + s, stage]
                        )

            u_hat = hard_bits()
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        u_hat = hard_bits()
        return u_hat, num_iters
