"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _boxplus_min_sum(a, b, alpha=0.9375):
    """min-sum 近似的 box-plus 运算"""
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


def _stage_pairs(N, stage):
    """第 stage 层（0=最靠近信源）的蝶形配对索引"""
    stride = 1 << stage
    ind1 = []
    ind2 = []
    for base in range(0, N, 2 * stride):
        for off in range(stride):
            ind1.append(base + off)
            ind2.append(base + off + stride)
    return np.asarray(ind1, dtype=int), np.asarray(ind2, dtype=int)


class BPDecoder:
    """BP 译码器（stage 因子图；信道 LLR 按比特倒序映射到因子图节点）"""

    LARGE = 1e7

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.info_idx = np.where(self.frozen_bits == 0)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.brp = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_orig = llr_ch.copy()
        n = self.n
        N = self.N
        ch = llr_orig[self.brp]

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        for iteration in range(self.max_iter):
            L.fill(0.0)
            R.fill(0.0)
            L[:, n] = ch
            R[:, 0] = 0.0
            R[self.frozen_idx, 0] = self.LARGE

            for stage in range(n - 1, -1, -1):
                ind1, ind2 = _stage_pairs(N, stage)
                for j, j2 in zip(ind1, ind2):
                    L[j, stage] = _boxplus_min_sum(
                        L[j, stage + 1],
                        L[j2, stage + 1] + R[j2, stage],
                        self.alpha,
                    )
                    L[j2, stage] = _boxplus_min_sum(
                        R[j, stage],
                        L[j, stage + 1],
                        self.alpha,
                    ) + L[j2, stage + 1]

            for stage in range(n):
                ind1, ind2 = _stage_pairs(N, stage)
                for j, j2 in zip(ind1, ind2):
                    R[j, stage + 1] = _boxplus_min_sum(
                        R[j, stage],
                        L[j2, stage + 1] + R[j2, stage],
                        self.alpha,
                    )
                    R[j2, stage + 1] = _boxplus_min_sum(
                        R[j, stage],
                        L[j, stage + 1],
                        self.alpha,
                    ) + R[j2, stage]

            total = L[:, 0] + R[:, 0]
            u_hat.fill(0)
            u_hat[self.info_idx] = (total[self.info_idx] < 0).astype(int)

            x_hat = polar_encode(u_hat)
            hard = (llr_orig < 0).astype(int)
            if np.array_equal(x_hat, hard):
                num_iters = iteration + 1
                break
            num_iters = iteration + 1

        total = L[:, 0] + R[:, 0]
        u_hat.fill(0)
        u_hat[self.info_idx] = (total[self.info_idx] < 0).astype(int)
        return u_hat, num_iters
