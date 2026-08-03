"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode


def _f_min_sum(x, y, alpha):
    sx = 0.0 if x == 0 else np.sign(x)
    sy = 0.0 if y == 0 else np.sign(y)
    return alpha * sx * sy * min(abs(x), abs(y))


def _f_boxplus(a, b):
    if abs(a) > 30:
        return b
    if abs(b) > 30:
        return a
    ta = np.tanh(a / 2.0)
    tb = np.tanh(b / 2.0)
    prod = np.clip(ta * tb, -0.999999, 0.999999)
    return 2.0 * np.arctanh(prod)


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375, use_boxplus=False):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.f_func = _f_boxplus if use_boxplus else lambda x, y: _f_min_sum(x, y, alpha)
        self.LARGE = 1e8

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N
        f_op = self.f_func

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                updated = np.zeros(N, dtype=bool)
                for i in range(0, N, 2 * s):
                    L[i, j - 1] = f_op(R[i, j] + L[i + s, j], L[i, j])
                    L[i + s, j - 1] = f_op(R[i, j], L[i, j]) + L[i + s, j]
                    updated[i] = True
                    updated[i + s] = True
                for i in range(N):
                    if not updated[i]:
                        L[i, j - 1] = L[i, j]

            for j in range(0, n):
                s = 1 << j
                updated = np.zeros(N, dtype=bool)
                for i in range(0, N, 2 * s):
                    R[i, j + 1] = f_op(R[i + s, j] + L[i + s, j + 1], R[i, j])
                    R[i + s, j + 1] = f_op(R[i, j], L[i + s, j + 1]) + R[i + s, j]
                    updated[i] = True
                    updated[i + s] = True
                for i in range(N):
                    if not updated[i]:
                        R[i, j + 1] = R[i, j]

            L[:, n] = llr_ch
            R[:, 0] = 0.0
            R[self.frozen_bits, 0] = self.LARGE

            for i in range(N):
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break

        for i in range(N):
            u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1
        u_hat[self.frozen_bits] = 0

        return u_hat, num_iters
