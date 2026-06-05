"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
参考：对称因子图上 L/R 双向消息传递（min-sum PE）
"""
import numpy as np
from encoder import polar_encode


def _boxplus(a, b, alpha=0.9375, use_minsum=True):
    """f/g 运算：min-sum 或 tanh 精确 box-plus"""
    if use_minsum:
        return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))
    a = np.clip(a, -30.0, 30.0)
    b = np.clip(b, -30.0, 30.0)
    return 2.0 * np.arctanh(np.tanh(a / 2.0) * np.tanh(b / 2.0))


class BPDecoder:
    """BP 译码器（因子图列 0..n，列 n 为信道 LLR）"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375, use_minsum=True):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.use_minsum = use_minsum
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N
        alpha = self.alpha
        use_ms = self.use_minsum
        g = lambda a, b: _boxplus(a, b, alpha, use_ms)

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            Ln = L.copy()
            Rn = R.copy()

            for col in range(n - 1, -1, -1):
                block = 1 << col
                for base in range(0, N, 2 * block):
                    for t in range(block):
                        j = base + t
                        Ln[j, col] = g(
                            L[j, col + 1],
                            L[j + block, col + 1] + R[j + block, col],
                        )
                        Ln[j + block, col] = (
                            g(L[j, col + 1], R[j, col]) + L[j + block, col + 1]
                        )

            for col in range(0, n):
                block = 1 << col
                for base in range(0, N, 2 * block):
                    for t in range(block):
                        j = base + t
                        Rn[j, col + 1] = g(
                            R[j + block, col + 1],
                            L[j + block, col + 1] + R[j + block, col],
                        )
                        Rn[j + block, col + 1] = (
                            g(L[j, col + 1], R[j, col]) + R[j + block, col]
                        )

            damp = 0.75
            L = (1.0 - damp) * L + damp * Ln
            R = (1.0 - damp) * R + damp * Rn
            L[:, n] = llr_ch
            R[:, 0] = 0.0
            R[self.frozen_idx, 0] = self.LARGE

            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            total = L[i, 0] + R[i, 0]
            u_hat[i] = 0 if total >= 0 else 1
        u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
