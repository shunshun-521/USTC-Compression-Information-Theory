"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode


def _f_min_sum(a, b, alpha):
    """min-sum 近似 f 函数"""
    if np.isscalar(a):
        return alpha * np.sign(a) * np.sign(b) * min(abs(a), abs(b))
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        LARGE = 1e7

        # L[i,j]: 右向 LLR 消息；R[i,j]: 左向 LLR 消息
        # j=0 为信源端，j=n 为信道端
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=np.int8)

        for it in range(1, self.max_iter + 1):
            # 从信道向信源更新 L 消息
            for layer in range(n, 0, -1):
                stride = 1 << (layer - 1)
                for block in range(0, N, stride << 1):
                    for t in range(stride):
                        i = block + t
                        j = i + stride
                        L[i, layer - 1] = _f_min_sum(
                            R[i, layer] + L[j, layer],
                            L[i, layer],
                            self.alpha,
                        )
                        L[j, layer - 1] = _f_min_sum(
                            R[i, layer],
                            L[i, layer],
                            self.alpha,
                        ) + L[j, layer]

            # 从信源向信道更新 R 消息
            for layer in range(1, n + 1):
                stride = 1 << (layer - 1)
                for block in range(0, N, stride << 1):
                    for t in range(stride):
                        i = block + t
                        j = i + stride
                        if layer < n:
                            R[i, layer] = _f_min_sum(
                                R[j, layer] + L[j, layer + 1],
                                R[i, layer - 1],
                                self.alpha,
                            )
                            R[j, layer] = _f_min_sum(
                                R[i, layer - 1],
                                L[i, layer + 1],
                                self.alpha,
                            ) + R[j, layer - 1]
                        else:
                            R[i, layer] = _f_min_sum(
                                R[j, layer] + L[j, layer],
                                R[i, layer - 1],
                                self.alpha,
                            )
                            R[j, layer] = _f_min_sum(
                                R[i, layer - 1],
                                L[i, layer],
                                self.alpha,
                            ) + R[j, layer - 1]

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(np.int8)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(np.int8)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
            num_iters = it

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(np.int8)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
