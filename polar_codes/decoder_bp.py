"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode, bit_reversal_permutation


def _f_min_sum(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        if self.frozen_bits.dtype != bool:
            self.frozen_bits = self.frozen_bits.astype(bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        brp = bit_reversal_permutation(self.N)
        llr_ch = llr_ch[brp]
        N = self.N
        n = self.n
        alpha = self.alpha

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)

        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_bits] = self.LARGE
        R[n, :] = 0.0

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(self.max_iter):
            # 左向传播 L（阶段 n -> 1）
            for stage in range(n, 0, -1):
                step = 1 << (stage - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx = i + k
                        L[stage - 1, idx] = _f_min_sum(
                            R[stage, idx] + L[stage, idx + step],
                            L[stage, idx], alpha)
                        L[stage - 1, idx + step] = (
                            _f_min_sum(R[stage, idx], L[stage, idx], alpha)
                            + L[stage, idx + step])

            # 右向传播 R（阶段 1 -> n）
            for stage in range(1, n + 1):
                step = 1 << (stage - 1)
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        idx = i + k
                        R[stage, idx] = _f_min_sum(
                            R[stage - 1, idx + step] + L[stage, idx + step],
                            R[stage - 1, idx], alpha)
                        R[stage, idx + step] = (
                            _f_min_sum(R[stage - 1, idx], L[stage, idx], alpha)
                            + R[stage - 1, idx + step])

            num_iters = it + 1

            for i in range(N):
                total = L[0, i] + R[0, i]
                u_hat[i] = 0 if total >= 0 else 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            total = L[0, i] + R[0, i]
            u_hat[i] = 0 if total >= 0 else 1
        u_hat[self.frozen_bits] = 0

        return u_hat, num_iters
