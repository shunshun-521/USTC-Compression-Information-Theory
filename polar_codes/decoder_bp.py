"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from decoder_sc import f_operation, _permute_channel_llr
from encoder import polar_encode


def _stage_masks(N):
    n = int(math.log2(N))
    masks = []
    for i in range(n):
        step = 1 << (n - i - 1)
        masks.append(np.arange(0, N, 2 * step, dtype=int))
    return masks


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.masks = _stage_masks(N)

    def _cn(self, a, b):
        return self.alpha * f_operation(a, b)

    def _update_left(self, R, L):
        for i in range(self.n - 1, -1, -1):
            add_k = 1 << (self.n - i - 1)
            for idx in self.masks[i]:
                L[i, idx] = self._cn(
                    L[i + 1, idx], L[i + 1, idx + add_k] + R[i, idx + add_k]
                )
                L[i, idx + add_k] = self._cn(
                    R[i, idx], L[i + 1, idx]
                ) + L[i + 1, idx + add_k]
        return L

    def _update_right(self, R, L):
        for i in range(self.n):
            add_k = 1 << (self.n - i - 1)
            for idx in self.masks[i]:
                R[i + 1, idx + add_k] = self._cn(
                    R[i, idx], L[i + 1, idx + add_k] + R[i, idx + add_k]
                )
                R[i + 1, idx] = self._cn(
                    R[i, idx], L[i + 1, idx]
                ) + R[i, idx + add_k]
        return R

    def decode(self, llr_ch):
        llr_orig = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = _permute_channel_llr(llr_orig, self.N)
        n = self.n
        N = self.N

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_ch
        R[0, self.frozen_idx] = self.LARGE

        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            L[n, :] = llr_ch
            R = self._update_right(R, L)
            L = self._update_left(R, L)

            for i in range(N):
                total = L[0, i] + R[0, i]
                u_hat[i] = 0 if (self.frozen_bits[i] or total >= 0) else 1

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, (llr_orig < 0).astype(int)):
                return u_hat, it

        return u_hat, self.max_iter
