"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _index_matrix(N):
    """各阶段上层节点索引（与 Kaira/Arikan 因子图一致）"""
    m = int(math.log2(N))
    masks = []
    for stage in range(m):
        block = 1 << (m - stage)
        half = block // 2
        idx = []
        for base in range(0, N, block):
            idx.extend(range(base, base + half))
        masks.append(np.array(idx, dtype=int))
    return masks


def _ms_check(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """
    BP 译码器（因子图 L/R 消息，min-sum，早停）。
    """

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.m = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.masks = _index_matrix(N)

    def _update_left(self, L, R):
        m = self.m
        N = self.N
        alpha = self.alpha
        for i in range(m - 1, -1, -1):
            add_k = N // (2 ** (i + 1))
            mask = self.masks[i]
            if len(mask) == 0:
                continue
            upper = mask
            lower = mask + add_k
            L[upper, i] = _ms_check(
                L[upper, i + 1], L[lower, i + 1] + R[lower, i], alpha
            )
            L[lower, i] = _ms_check(R[upper, i], L[upper, i + 1], alpha) + L[
                lower, i + 1
            ]

    def _update_right(self, L, R):
        m = self.m
        N = self.N
        alpha = self.alpha
        for i in range(m):
            add_k = N // (2 ** (i + 1))
            mask = self.masks[i]
            if len(mask) == 0:
                continue
            upper = mask
            lower = mask + add_k
            R[upper, i + 1] = _ms_check(
                R[upper, i], L[lower, i + 1] + R[lower, i], alpha
            )
            R[lower, i + 1] = _ms_check(R[upper, i], L[upper, i + 1], alpha) + R[
                lower, i
            ]

    def _decision(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(self.N, dtype=int)
        u_hat[total < 0] = 1
        u_hat[self.frozen_bits] = 0
        return u_hat

    def decode(self, llr_ch):
        """
        主译码函数。
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        m = self.m
        N = self.N

        L = np.zeros((N, m + 1), dtype=np.float64)
        R = np.zeros((N, m + 1), dtype=np.float64)
        L[:, m] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        hard_ch = (llr_ch < 0).astype(int)
        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            self._update_left(L, R)
            self._update_right(L, R)

            u_hat = self._decision(L, R)
            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        u_hat = self._decision(L, R)
        return u_hat, num_iters
