"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    """min-sum 近似 f 函数"""
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 (u_hat, num_iters)
        """
        N = self.N
        n = self.n
        alpha = self.alpha
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        # L[i][j]: 从右到左的消息，R[i][j]: 从左到右的消息
        # 使用 shape (N, n+1) 数组
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            # 从右到左更新 L（stage n-1 到 0）
            for stage in range(n - 1, -1, -1):
                s = 1 << stage
                for i in range(0, N, 2 * s):
                    for j in range(s):
                        idx_u = i + j
                        idx_l = i + j + s
                        L[idx_u, stage] = _f_min_sum(
                            R[idx_u, stage] + L[idx_l, stage + 1],
                            L[idx_u, stage + 1],
                            alpha,
                        )
                        L[idx_l, stage] = (
                            _f_min_sum(R[idx_u, stage], L[idx_u, stage + 1], alpha)
                            + L[idx_l, stage + 1]
                        )

            # 从左到右更新 R（stage 0 到 n-1）
            for stage in range(n):
                s = 1 << stage
                for i in range(0, N, 2 * s):
                    for j in range(s):
                        idx_u = i + j
                        idx_l = i + j + s
                        R[idx_u, stage + 1] = _f_min_sum(
                            R[idx_l, stage] + L[idx_l, stage + 1],
                            R[idx_u, stage],
                            alpha,
                        )
                        R[idx_l, stage + 1] = (
                            _f_min_sum(R[idx_u, stage], L[idx_u, stage + 1], alpha)
                            + R[idx_l, stage]
                        )

            # 早停检查
            total_llr = L[:, 0] + R[:, 0]
            u_hat = (total_llr < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
        else:
            total_llr = L[:, 0] + R[:, 0]
            u_hat = (total_llr < 0).astype(int)
            u_hat[self.frozen_bits] = 0
            num_iters = self.max_iter

        return u_hat, num_iters
