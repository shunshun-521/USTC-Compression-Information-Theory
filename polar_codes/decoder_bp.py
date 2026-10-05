"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import os
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


def _sign_pm(x):
    s = np.sign(x)
    return np.where(s == 0, 1.0, s)


def _minsum(a, b, alpha):
    return alpha * _sign_pm(a) * _sign_pm(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（极化因子图）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        # L[i, j]: 从右向左；R[i, j]: 从左向右（i 为子块起始，j 为层）
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self._large

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            # 右到左更新 L
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, s * 2):
                    for k in range(s):
                        idx = i + k
                        L[idx, j - 1] = _minsum(
                            R[idx, j - 1] + L[idx + s, j],
                            L[idx, j],
                            self.alpha,
                        )
                        L[idx + s, j - 1] = _minsum(
                            R[idx, j - 1], L[idx, j], self.alpha
                        ) + L[idx + s, j]

            # 左到右更新 R
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, s * 2):
                    for k in range(s):
                        idx = i + k
                        R[idx, j + 1] = _minsum(
                            R[idx + s, j] + L[idx + s, j + 1],
                            R[idx, j],
                            self.alpha,
                        )
                        R[idx + s, j + 1] = _minsum(
                            R[idx, j], L[idx, j + 1], self.alpha
                        ) + R[idx + s, j]

            # 判决与早停
            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            if os.environ.get("POLAR_BP_EARLY_STOP", "0") == "1":
                x_hat = polar_encode(u_hat)
                hard_ch = (llr_ch < 0).astype(np.int8)
                if np.array_equal(x_hat, hard_ch):
                    break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
