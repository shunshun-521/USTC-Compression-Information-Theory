"""
极化码 BP（置信传播）译码器
基于因子图 min-sum，含早停
"""
import math
import numpy as np
from encoder import polar_encode


def _f_min_sum(a, b, alpha=0.9375):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """分层 BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e8

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        alpha = self.alpha

        # L[s, i]: 左向 LLR 消息，层 s=0..n
        L = np.zeros((n + 1, N), dtype=np.float64)
        # R[s, i]: 右向 LLR 消息
        R = np.zeros((n + 1, N), dtype=np.float64)

        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_idx] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # 右向消息更新（s: 0 -> n-1）
            for s in range(n):
                block = 2 ** (s + 1)
                half = block // 2
                for i in range(0, N, block):
                    for j in range(half):
                        idx = i + j
                        R[s + 1, idx] = _f_min_sum(
                            R[s, idx], R[s, idx + half] + L[s + 1, idx + half], alpha
                        )
                        R[s + 1, idx + half] = _f_min_sum(
                            R[s, idx], L[s + 1, idx], alpha
                        ) + R[s, idx + half]

            # 左向消息更新（s: n-1 -> 0）
            for s in range(n - 1, -1, -1):
                block = 2 ** (s + 1)
                half = block // 2
                for i in range(0, N, block):
                    for j in range(half):
                        idx = i + j
                        L[s, idx] = _f_min_sum(
                            L[s + 1, idx], L[s + 1, idx + half] + R[s, idx + half], alpha
                        )
                        L[s, idx + half] = _f_min_sum(
                            R[s, idx], L[s + 1, idx], alpha
                        ) + L[s + 1, idx + half]

            num_iters = it

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    total = L[0, i] + R[0, i]
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                total = L[0, i] + R[0, i]
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
