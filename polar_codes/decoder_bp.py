"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def _hard_decode(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total[i] >= 0 else 1
        return u_hat

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        N = self.N
        n = self.n
        alpha = self.alpha
        llr_ch = -np.asarray(llr_ch, dtype=np.float64)

        # 列 0..n，列 n 为信道端
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            # 从右到左更新 L
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                L_left = L[:, j]
                L_right = L[:, j].copy()
                for block in range(0, N, 2 * s):
                    for i in range(block, block + s):
                        ip = i + s
                        L_left[i] = _f_min_sum(
                            R[i, j] + L[ip, j], L[i, j], alpha
                        )
                        L_right[ip] = (
                            _f_min_sum(R[i, j], L[i, j], alpha) + L[ip, j]
                        )
                L[:, j - 1] = L_left
                for block in range(0, N, 2 * s):
                    for i in range(block + s, block + 2 * s):
                        L[i, j - 1] = L_right[i]

            # 从左到右更新 R
            for j in range(0, n):
                s = 1 << j
                R_left = R[:, j].copy()
                R_right = R[:, j].copy()
                for block in range(0, N, 2 * s):
                    for i in range(block, block + s):
                        ip = i + s
                        R_left[i] = _f_min_sum(
                            R[ip, j] + L[ip, j + 1], R[i, j], alpha
                        )
                        R_right[ip] = (
                            _f_min_sum(R[i, j], L[i, j + 1], alpha) + R[ip, j]
                        )
                R[:, j + 1] = R_right
                for block in range(0, N, 2 * s):
                    for i in range(block, block + s):
                        R[i, j + 1] = R_left[i]

            u_hat = self._hard_decode(L, R)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch > 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                return u_hat, num_iters

        return self._hard_decode(L, R), num_iters
