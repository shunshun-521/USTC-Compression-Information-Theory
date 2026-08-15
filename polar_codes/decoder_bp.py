"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import _prepare_llr
from encoder import polar_encode


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _g(self, a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        sign = np.sign(a) * np.sign(b)
        sign = np.where(sign == 0, 1, sign)
        return self.alpha * sign * np.minimum(np.abs(a), np.abs(b))

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        llr_ch = _prepare_llr(llr_ch)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # 右到左更新 L
            for j in range(n - 1, -1, -1):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        iu = i + k
                        il = i + k + step
                        L[iu, j] = self._g(L[iu, j + 1], R[il, j] + L[il, j + 1])
                        L[il, j] = self._g(L[iu, j + 1], R[iu, j]) + L[il, j + 1]

            # 左到右更新 R
            for j in range(0, n):
                step = 1 << j
                for i in range(0, N, 2 * step):
                    for k in range(step):
                        iu = i + k
                        il = i + k + step
                        R[iu, j + 1] = self._g(R[iu, j], L[il, j + 1] + R[il, j])
                        R[il, j + 1] = self._g(R[iu, j], L[iu, j + 1]) + R[il, j]

            total_u = L[:, 0] + R[:, 0]
            u_hat = np.where(total_u >= 0, 0, 1).astype(int)
            u_hat[self.frozen_bits] = 0

            total_x = L[:, n] + R[:, n]
            x_hat = np.where(total_x >= 0, 0, 1).astype(int)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total_u = L[:, 0] + R[:, 0]
        u_hat = np.where(total_u >= 0, 0, 1).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
