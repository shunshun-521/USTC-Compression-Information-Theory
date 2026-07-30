"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from decoder_sc import f_operation, g_operation
from encoder import polar_encode, bit_reversal_permutation


def _minsum_f(a, b, alpha):
    """Min-sum f；0 表示无先验消息。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    result = alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))
    result = np.where(np.abs(a) < 1e-12, alpha * b, result)
    result = np.where(np.abs(b) < 1e-12, alpha * a, result)
    return result


class BPDecoder:
    """BP 译码器（因子图蝶形结构与 SC 一致）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.LARGE = 1e6

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, num_iters。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        rev = bit_reversal_permutation(N)

        # L[layer][i]: 从左到右消息; R[layer][i]: 从右到左消息
        L = [np.zeros(N, dtype=np.float64) for _ in range(n + 1)]
        R = [np.zeros(N, dtype=np.float64) for _ in range(n + 1)]
        L[n][:] = llr_ch[rev]
        R[0][:] = 0.0
        R[0][self.frozen_idx] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # 右到左：更新 L 消息
            for layer in range(n, 0, -1):
                block = 1 << (layer - 1)
                for start in range(0, N, 2 * block):
                    left = slice(start, start + block)
                    right = slice(start + block, start + 2 * block)
                    L[layer - 1][left] = _minsum_f(
                        R[layer][left] + L[layer][right],
                        L[layer][left],
                        self.alpha,
                    )
                    L[layer - 1][right] = _minsum_f(
                        R[layer][left], L[layer][left], self.alpha
                    ) + L[layer][right]

            # 左到右：更新 R 消息
            for layer in range(0, n):
                block = 1 << layer
                for start in range(0, N, 2 * block):
                    left = slice(start, start + block)
                    right = slice(start + block, start + 2 * block)
                    R[layer + 1][left] = _minsum_f(
                        R[layer][right] + L[layer + 1][right],
                        R[layer][left],
                        self.alpha,
                    )
                    R[layer + 1][right] = _minsum_f(
                        R[layer][left], L[layer + 1][left], self.alpha
                    ) + R[layer][right]

            total = L[0] + R[0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[0] + R[0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
