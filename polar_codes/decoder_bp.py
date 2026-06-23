"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode


def _bp_f(x, y, alpha):
  """min-sum f with normalization."""
  return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def decode(self, llr_ch):
        """
        主译码函数。

        返回：
            u_hat: 长度 N 的估计源序列
            num_iters: 实际迭代次数
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            # 右到左更新 L
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    L[i:i + s, j - 1] = _bp_f(
                        R[i:i + s, j - 1] + L[i + s:i + 2 * s, j],
                        L[i:i + s, j],
                        self.alpha,
                    )
                    L[i + s:i + 2 * s, j - 1] = _bp_f(
                        R[i:i + s, j - 1],
                        L[i:i + s, j],
                        self.alpha,
                    ) + L[i + s:i + 2 * s, j]

            # 左到右更新 R
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    R[i:i + s, j + 1] = _bp_f(
                        R[i + s:i + 2 * s, j] + L[i + s:i + 2 * s, j + 1],
                        R[i:i + s, j],
                        self.alpha,
                    )
                    R[i + s:i + 2 * s, j + 1] = _bp_f(
                        R[i:i + s, j],
                        L[i:i + s, j + 1],
                        self.alpha,
                    ) + R[i + s:i + 2 * s, j]

            # 早停检查
            total = L[:, 0] + R[:, 0]
            u_hat = np.where(total >= 0, 0, 1).astype(np.int8)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
            num_iters = it
        else:
            total = L[:, 0] + R[:, 0]
            u_hat = np.where(total >= 0, 0, 1).astype(np.int8)
            u_hat[self.frozen_bits] = 0

        return u_hat, num_iters
