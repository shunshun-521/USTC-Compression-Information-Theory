"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _ms_boxplus(x, y, alpha):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)

    def _prepare_llr(self, llr_ch):
        """信道 LLR 按比特倒序对齐因子图右侧。"""
        return np.asarray(llr_ch, dtype=np.float64)[self.br].copy()

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = self._prepare_llr(llr_ch)
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                stride = 1 << (j - 1)
                for block in range(0, N, stride << 1):
                    left = block
                    right = block + stride
                    for i in range(left, right):
                        L[i, j - 1] = _ms_boxplus(
                            R[i, j] + L[i + stride, j], L[i, j], self.alpha
                        )
                        L[i + stride, j - 1] = _ms_boxplus(
                            R[i, j], L[i, j], self.alpha
                        ) + L[i + stride, j]

            for j in range(0, n):
                stride = 1 << j
                for block in range(0, N, stride << 1):
                    left = block
                    right = block + stride
                    for i in range(left, right):
                        R[i, j + 1] = _ms_boxplus(
                            R[i + stride, j] + L[i + stride, j + 1],
                            R[i, j],
                            self.alpha,
                        )
                        R[i + stride, j + 1] = _ms_boxplus(
                            R[i, j], L[i, j + 1], self.alpha
                        ) + R[i + stride, j]

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    total = L[i, 0] + R[i, 0]
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
            num_iters = it

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat.astype(int), num_iters
