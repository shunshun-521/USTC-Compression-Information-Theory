"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
消息 L/R 形状为 (N, n+1)，列 0=信源端，列 n=信道端
"""
import math
import numpy as np
from decoder_sc import _frozen_mask, bit_reversed
from encoder import polar_encode


def _min_sum(x, y, alpha=1.0):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = _frozen_mask(frozen_bits)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]
        self.LARGE = 1e6

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        alpha = self.alpha

        # 编码含比特倒序，信道 LLR 需对齐至蝶形域
        rev = np.array([bit_reversed(i, n) for i in range(N)])
        llr = llr_ch[rev]

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # 从右到左更新 L（列 n -> 1）
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_i = i + k
                        idx_j = i + k + s
                        L[idx_i, j - 1] = _min_sum(
                            R[idx_i, j] + L[idx_j, j], L[idx_i, j], alpha
                        )
                        L[idx_j, j - 1] = _min_sum(
                            R[idx_i, j], L[idx_i, j], alpha
                        ) + L[idx_j, j]

            # 从左到右更新 R（列 0 -> n-1）
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_i = i + k
                        idx_j = i + k + s
                        R[idx_i, j + 1] = _min_sum(
                            R[idx_j, j] + L[idx_j, j + 1], R[idx_i, j], alpha
                        )
                        R[idx_j, j + 1] = _min_sum(
                            R[idx_i, j], L[idx_i, j + 1], alpha
                        ) + R[idx_j, j]

            total = L[:, 0] + R[:, 0]
            u_hat[self.info_idx] = (total[self.info_idx] < 0).astype(int)
            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break
        else:
            total = L[:, 0] + R[:, 0]
            u_hat[self.info_idx] = (total[self.info_idx] < 0).astype(int)

        return u_hat, num_iters
