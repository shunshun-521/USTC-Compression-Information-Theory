"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _boxplus_min_sum(a, b, alpha=0.9375):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self._build_stage_indices()

    def _build_stage_indices(self):
        self.stage_pairs = []
        for stage in range(1, self.n + 1):
            stride = 1 << (stage - 1)
            pairs = []
            for i in range(0, self.N, stride << 1):
                for j in range(stride):
                    pairs.append((i + j, i + j + stride))
            self.stage_pairs.append(pairs)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        L = np.zeros((self.N, self.n + 1), dtype=np.float64)
        R = np.zeros((self.N, self.n + 1), dtype=np.float64)
        L[:, self.n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = 1e6

        num_iters = self.max_iter
        u_hat = np.zeros(self.N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for stage in range(self.n, 0, -1):
                pairs = self.stage_pairs[stage - 1]
                for i, ip in pairs:
                    L[i, stage - 1] = _boxplus_min_sum(
                        R[i, stage - 1] + L[ip, stage],
                        L[i, stage],
                        self.alpha,
                    )
                    L[ip, stage - 1] = _boxplus_min_sum(
                        R[i, stage - 1], L[i, stage], self.alpha
                    ) + L[ip, stage]

            for stage in range(1, self.n + 1):
                pairs = self.stage_pairs[stage - 1]
                for i, ip in pairs:
                    R[i, stage] = _boxplus_min_sum(
                        R[ip, stage - 1] + L[ip, stage],
                        R[i, stage - 1],
                        self.alpha,
                    )
                    R[ip, stage] = _boxplus_min_sum(
                        R[i, stage - 1], L[i, stage], self.alpha
                    ) + R[ip, stage - 1]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
