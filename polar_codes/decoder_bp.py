"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import _logdomain_sum


def _g_box_exact(x, y):
    """精确 log-domain G 函数（f 运算）"""
    return _logdomain_sum(x + y, 0.0) - _logdomain_sum(x, y)


def _g_box(x, y, alpha):
    if np.isinf(x) and not np.isinf(y):
        return y
    if not np.isinf(x) and np.isinf(y):
        return x
    if np.isinf(x) and np.isinf(y):
        return x
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_mask = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e10

    def decode(self, llr_ch):
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_mask, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(self.max_iter):
            num_iters = it + 1
            L[:, n] = llr_ch

            for j in range(n - 1, -1, -1):
                half = 1 << j
                block = half << 1
                for block_start in range(0, N, block):
                    for k in range(half):
                        i = block_start + k
                        L[i, j] = _g_box(
                            L[i, j + 1],
                            L[i + half, j + 1] + R[i + half, j],
                            self.alpha,
                        )
                        L[i + half, j] = (
                            _g_box(R[i, j], L[i, j + 1], self.alpha)
                            + L[i + half, j + 1]
                        )

            for j in range(n):
                half = 1 << j
                block = half << 1
                for block_start in range(0, N, block):
                    for k in range(half):
                        i = block_start + k
                        R[i, j + 1] = _g_box(
                            R[i, j],
                            L[i + half, j + 1] + R[i + half, j],
                            self.alpha,
                        )
                        R[i + half, j + 1] = (
                            _g_box(R[i, j], L[i, j + 1], self.alpha)
                            + R[i + half, j]
                        )

            R[self.frozen_mask, 0] = self.LARGE

            for i in range(N):
                if self.frozen_mask[i]:
                    u_hat[i] = 0
                else:
                    total = L[i, 0] + R[i, 0]
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            if self.frozen_mask[i]:
                u_hat[i] = 0
            else:
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
