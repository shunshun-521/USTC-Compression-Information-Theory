"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode


LARGE = 1e6


def _build_stage_pairs(N):
    """预计算各 stage 的 (top, bottom) 节点索引对"""
    n = int(math.log2(N))
    left_pairs = []
    right_pairs = []
    for stage in range(n):
        stride = 1 << stage
        block = 2 * stride
        tops = []
        bottoms = []
        for base in range(0, N, block):
            for k in range(stride):
                tops.append(base + k)
                bottoms.append(base + k + stride)
        tops = np.array(tops, dtype=np.int64)
        bottoms = np.array(bottoms, dtype=np.int64)
        left_pairs.append((tops, bottoms))
        right_pairs.append((tops, bottoms))
    return left_pairs, right_pairs


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self._left_pairs, self._right_pairs = _build_stage_pairs(N)

    def _f_min_sum(self, x, y):
        return self.alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))

    def decode(self, llr_ch):
        """主译码函数"""
        N = self.N
        n = self.n
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for iteration in range(self.max_iter):
            num_iters = iteration + 1

            for stage in range(n - 1, -1, -1):
                tops, bottoms = self._left_pairs[stage]
                L[tops, stage] = self._f_min_sum(
                    L[tops, stage + 1],
                    R[bottoms, stage] + L[bottoms, stage + 1],
                )
                L[bottoms, stage] = (
                    self._f_min_sum(L[tops, stage + 1], R[tops, stage])
                    + L[bottoms, stage + 1]
                )

            for stage in range(0, n):
                tops, bottoms = self._right_pairs[stage]
                R[tops, stage + 1] = self._f_min_sum(
                    R[tops, stage],
                    R[bottoms, stage] + L[bottoms, stage + 1],
                )
                R[bottoms, stage + 1] = (
                    self._f_min_sum(R[tops, stage], L[tops, stage + 1])
                    + R[bottoms, stage]
                )

            posterior = L[:, 0] + R[:, 0]
            u_hat = (posterior < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_decision = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_decision):
                break

        u_hat[self.frozen_bits] = 0
        posterior = L[:, 0] + R[:, 0]
        u_hat[~self.frozen_bits] = (posterior[~self.frozen_bits] < 0).astype(int)

        return u_hat, num_iters
