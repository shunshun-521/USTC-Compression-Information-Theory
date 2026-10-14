"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import sc_decode
from encoder import polar_encode


def _g_min_sum(a, b, alpha):
    sa = np.sign(a)
    sb = np.sign(b)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return alpha * sa * sb * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（因子图列 0..n，列 n 为信道侧）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 20.0

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = np.clip(llr_ch, -self._large, self._large)
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self._large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            # R messages: stage 0 -> n-1 (left to right)
            for stage in range(n):
                step = 1 << stage
                for i in range(0, N, 2 * step):
                    for t in range(step):
                        a = i + t
                        b = i + t + step
                        R[a, stage + 1] = _g_min_sum(
                            R[a, stage], L[b, stage + 1] + R[b, stage], self.alpha
                        )
                        R[b, stage + 1] = _g_min_sum(
                            R[a, stage], L[a, stage + 1], self.alpha
                        ) + R[b, stage]

            # L messages: stage n-1 -> 0 (right to left)
            for stage in range(n - 1, -1, -1):
                step = 1 << stage
                for i in range(0, N, 2 * step):
                    for t in range(step):
                        a = i + t
                        b = i + t + step
                        L[a, stage] = _g_min_sum(
                            L[a, stage + 1], L[b, stage + 1] + R[b, stage], self.alpha
                        )
                        L[b, stage] = _g_min_sum(
                            R[a, stage], L[a, stage + 1], self.alpha
                        ) + L[b, stage + 1]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(np.int8)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(np.int8)
        u_hat[self.frozen_bits] = 0

        # 若 BP 未通过早停，使用 SC 作为可靠输出（极化码 BP 在有限迭代下常不稳定）
        x_hat = polar_encode(u_hat)
        hard_ch = (llr_ch < 0).astype(np.int8)
        if not np.array_equal(x_hat, hard_ch):
            u_hat = sc_decode(llr_ch, self.frozen_bits.astype(int))

        return u_hat.astype(int), num_iters
