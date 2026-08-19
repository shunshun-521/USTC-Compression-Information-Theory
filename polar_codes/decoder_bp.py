"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math
from decoder_sc import _sign_pm
from encoder import polar_encode, bit_reversal_permutation


class BPDecoder:
    """BP 译码器（极化码因子图 min-sum）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e7

    def _f_min_sum(self, a, b):
        return self.alpha * _sign_pm(a) * _sign_pm(b) * np.minimum(np.abs(a), np.abs(b))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        br = bit_reversal_permutation(N)
        llr_work = llr_ch[br].copy()

        # L[l][i]: 层 l（0=信源）到比特 i 的左向消息；层 n 为信道
        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_work

        for i in range(N):
            if self.frozen_bits[i]:
                R[0, i] = self.LARGE

        u_hat = np.zeros(N, dtype=int)
        num_iters = 0

        for it in range(self.max_iter):
            num_iters = it + 1

            # 右到左更新 L（层 n-1 到 0）
            for lam in range(n - 1, -1, -1):
                step = 2 ** lam
                half = step
                num_blocks = 2 ** (n - lam - 1)
                for block in range(num_blocks):
                    base = block * 2 * step
                    for phi in range(half):
                        i1 = base + phi
                        i2 = i1 + half
                        ra = R[lam, i1]
                        la = L[lam + 1, i1]
                        lb = L[lam + 1, i2]
                        L[lam, i1] = self._f_min_sum(ra + lb, la)
                        L[lam, i2] = self._f_min_sum(ra, la) + lb

            # 左到右更新 R（层 0 到 n-1）
            for lam in range(n):
                step = 2 ** lam
                half = step
                num_blocks = 2 ** (n - lam - 1)
                for block in range(num_blocks):
                    base = block * 2 * step
                    for phi in range(half):
                        i1 = base + phi
                        i2 = i1 + half
                        ra = R[lam, i1]
                        rb = R[lam, i2]
                        la = L[lam + 1, i1]
                        lb = L[lam + 1, i2]
                        R[lam, i2] = self._f_min_sum(rb + lb, ra)
                        R[lam, i1] = self._f_min_sum(ra, la) + rb

            for i in range(N):
                total = L[0, i] + R[0, i]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, (llr_ch < 0).astype(int)):
                break

        return u_hat, num_iters
