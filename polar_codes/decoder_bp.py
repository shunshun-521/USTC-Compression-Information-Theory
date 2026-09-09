"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _f_min_sum(x, y, alpha):
    """min-sum f 运算"""
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


def _g_sum(x, y):
    """g 运算（加法形式）"""
    return x + y


class BPDecoder:
    """
    BP 译码器。
    因子图 stage 0（左）到 stage n（右/信道）。
    L[s][i]: 从右向左的消息；R[s][i]: 从左向右的消息。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)

        L[n, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_bits] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # L 消息：从 stage n-1 到 0（右到左）
            for s in range(n - 1, -1, -1):
                half = 1 << s
                block = half << 1
                for p in range(0, N, block):
                    for o in range(half):
                        i = p + o
                        j = p + o + half
                        L[s, i] = _f_min_sum(
                            L[s + 1, i],
                            _g_sum(L[s + 1, j], R[s + 1, j]),
                            self.alpha,
                        )
                        L[s, j] = _g_sum(
                            _f_min_sum(R[s + 1, i], L[s + 1, i], self.alpha),
                            L[s + 1, j],
                        )

            # R 消息：从 stage 0 到 n-1（左到右）
            for s in range(0, n):
                half = 1 << s
                block = half << 1
                for p in range(0, N, block):
                    for o in range(half):
                        i = p + o
                        j = p + o + half
                        R[s + 1, i] = _f_min_sum(
                            R[s, i],
                            _g_sum(L[s + 1, j], R[s + 1, j]),
                            self.alpha,
                        )
                        R[s + 1, j] = _g_sum(
                            _f_min_sum(R[s, i], L[s + 1, i], self.alpha),
                            R[s + 1, j],
                        )

            num_iters = it

            # 早停检查
            for idx in range(N):
                if self.frozen_bits[idx]:
                    u_hat[idx] = 0
                else:
                    u_hat[idx] = 0 if (L[0, idx] + R[0, idx]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for idx in range(N):
            if self.frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if (L[0, idx] + R[0, idx]) >= 0 else 1

        return u_hat, num_iters
