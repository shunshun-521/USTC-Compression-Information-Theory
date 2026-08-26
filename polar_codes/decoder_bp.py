"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from channel import hard_decision_llr
from encoder import bit_reversal_permutation, polar_encode


def bp_f_operation(a, b, alpha):
    """min-sum 近似的 f 运算，带缩放因子 alpha。"""
    sa, sb = np.sign(a), np.sign(b)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return alpha * sa * sb * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e7

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_indices = np.where(self.frozen_bits == 1)[0]

    def _update_left(self, L, R):
        n, N = self.n, self.N
        for stage in range(n - 1, -1, -1):
            block = 1 << (stage + 1)
            half = block // 2
            for idx in range(N):
                block_start = (idx // block) * block
                if idx < block_start + half:
                    sibling = idx + half
                    L[idx, stage] = bp_f_operation(
                        R[idx, stage + 1] + L[sibling, stage + 1],
                        L[idx, stage + 1],
                        self.alpha,
                    )
                else:
                    sibling = idx - half
                    L[idx, stage] = bp_f_operation(
                        R[sibling, stage + 1],
                        L[sibling, stage + 1],
                        self.alpha,
                    ) + L[idx, stage + 1]

    def _update_right(self, L, R):
        n, N = self.n, self.N
        for stage in range(n):
            block = 1 << (stage + 1)
            half = block // 2
            for idx in range(N):
                block_start = (idx // block) * block
                if idx < block_start + half:
                    sibling = idx + half
                    R[idx, stage + 1] = bp_f_operation(
                        R[sibling, stage + 1] + L[sibling, stage + 1],
                        R[idx, stage],
                        self.alpha,
                    )
                else:
                    sibling = idx - half
                    R[idx, stage + 1] = bp_f_operation(
                        R[sibling, stage],
                        L[sibling, stage + 1],
                        self.alpha,
                    ) + R[idx, stage]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        br = bit_reversal_permutation(N)
        channel_llr = llr_ch[br].copy()

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = channel_llr
        R[:, 0] = 0.0
        R[self.frozen_indices, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            self._update_left(L, R)
            self._update_right(L, R)
            L[:, n] = channel_llr

            total_llr = L[:, 0] + R[:, 0]
            u_hat = (total_llr < 0).astype(int)
            u_hat[self.frozen_indices] = 0

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                break

        total_llr = L[:, 0] + R[:, 0]
        u_hat = (total_llr < 0).astype(int)
        u_hat[self.frozen_indices] = 0
        return u_hat, num_iters
