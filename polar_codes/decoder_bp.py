"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.LARGE = 1e6

    def _f_min_sum(self, x, y):
        """min-sum 近似的 f 运算"""
        return self.alpha * f_operation(x, y)

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

        # L[node, stage]: 从右向左的消息，stage=0 为信源端，stage=n 为信道端
        # R[node, stage]: 从左向右的消息
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        for it in range(self.max_iter):
            num_iters = it + 1

            # 从右到左更新 L 消息（stage n -> 1）
            for stage in range(n, 0, -1):
                s = 1 << (stage - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        L[idx, stage - 1] = self._f_min_sum(
                            R[idx, stage] + L[idx + s, stage],
                            L[idx, stage])
                        L[idx + s, stage - 1] = (
                            self._f_min_sum(R[idx, stage], L[idx, stage])
                            + L[idx + s, stage])

            # 从左到右更新 R 消息（stage 0 -> n-1）
            for stage in range(0, n):
                s = 1 << stage
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        R[idx, stage + 1] = self._f_min_sum(
                            R[idx + s, stage] + L[idx + s, stage + 1],
                            R[idx, stage])
                        R[idx + s, stage + 1] = (
                            self._f_min_sum(R[idx, stage], L[idx, stage + 1])
                            + R[idx + s, stage])

            # 早停检查
            total_llr = L[:, 0] + R[:, 0]
            u_hat = (total_llr < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                break

        total_llr = L[:, 0] + R[:, 0]
        u_hat = (total_llr < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
