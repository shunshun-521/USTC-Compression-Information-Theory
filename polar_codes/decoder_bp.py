"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode

_LARGE = 1e8


class BPDecoder:
    """BP 译码器（因子图 min-sum）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def _f_min_sum(self, a, b):
        """min-sum / box-plus；零消息时传递另一路。"""
        a, b = float(a), float(b)
        if abs(a) < 1e-12:
            return b
        if abs(b) < 1e-12:
            return a
        if abs(a) > 30 or abs(b) > 30:
            return self.alpha * f_operation(a, b)
        ta = np.tanh(a / 2.0)
        tb = np.tanh(b / 2.0)
        prod = ta * tb
        if abs(prod) >= 1.0 - 1e-12:
            return self.alpha * f_operation(a, b)
        return self.alpha * 2.0 * np.arctanh(prod)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # 右到左更新 L
            for j in range(n - 1, -1, -1):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    s = step
                    L[i, j] = self._f_min_sum(
                        R[i, j + 1] + L[i + s, j + 1], L[i, j + 1]
                    )
                    L[i + s, j] = (
                        self._f_min_sum(R[i, j + 1], L[i, j + 1])
                        + L[i + s, j + 1]
                    )

            # 左到右更新 R
            for j in range(1, n + 1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    s = step
                    R[i, j] = self._f_min_sum(
                        R[i + s, j] + L[i + s, j], R[i, j - 1]
                    )
                    R[i + s, j] = (
                        self._f_min_sum(R[i, j - 1], L[i, j])
                        + R[i + s, j - 1]
                    )

            L[self.frozen_bits, 0] = _LARGE

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        return u_hat, num_iters
