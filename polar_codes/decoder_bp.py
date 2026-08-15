"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode, bit_reversal_permutation


def _f_min_sum(x, y, alpha):
    """min-sum 近似的 f 运算。"""
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.rev = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        N = self.N
        n = self.n
        alpha = self.alpha

        llr = llr_ch[self.rev]

        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))
        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for stage in range(n - 1, -1, -1):
                bs = 1 << stage
                for i in range(0, N, 2 * bs):
                    for j in range(bs):
                        idx = i + j
                        L[idx, stage] = _f_min_sum(
                            L[idx, stage + 1],
                            L[idx + bs, stage + 1] + R[idx + bs, stage],
                            alpha,
                        )
                        L[idx + bs, stage] = _f_min_sum(
                            R[idx, stage], L[idx, stage + 1], alpha
                        ) + L[idx + bs, stage + 1]

            for stage in range(n):
                bs = 1 << stage
                for i in range(0, N, 2 * bs):
                    for j in range(bs):
                        idx = i + j
                        R[idx, stage + 1] = _f_min_sum(
                            R[idx + bs, stage] + L[idx + bs, stage + 1],
                            R[idx, stage],
                            alpha,
                        )
                        R[idx + bs, stage + 1] = _f_min_sum(
                            R[idx, stage], L[idx, stage + 1], alpha
                        ) + R[idx + bs, stage]

            total_llr = L[:, 0] + R[:, 0]
            u_hat = (total_llr < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

        total_llr = L[:, 0] + R[:, 0]
        u_hat = (total_llr < 0).astype(int)
        u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
