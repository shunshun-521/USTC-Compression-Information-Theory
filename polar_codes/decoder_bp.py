"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode


def _minsum_f(a, b, alpha):
    """min-sum 近似 f 运算，带归一化因子 alpha"""
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def decode(self, llr_ch, llr_ch_natural=None):
        """
        主译码函数。

        参数：
            llr_ch: 比特倒序后的译码器输入 LLR
            llr_ch_natural: 原始信道 LLR（早停校验用）

        返回：
            u_hat: 长度 N 的估计源序列
            num_iters: 实际迭代次数
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if llr_ch_natural is None:
            llr_ch_natural = llr_ch
        else:
            llr_ch_natural = np.asarray(llr_ch_natural, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen, 0] = self.large

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # 从右到左更新 L 消息
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_i = i + k
                        idx_is = i + k + s
                        L[idx_i, j - 1] = _minsum_f(
                            R[idx_i, j] + L[idx_is, j],
                            L[idx_i, j],
                            self.alpha,
                        )
                        L[idx_is, j - 1] = (
                            _minsum_f(R[idx_i, j], L[idx_i, j], self.alpha)
                            + L[idx_is, j]
                        )

            # 从左到右更新 R 消息
            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_i = i + k
                        idx_is = i + k + s
                        R[idx_i, j] = _minsum_f(
                            R[idx_is, j] + L[idx_is, j],
                            R[idx_i, j - 1],
                            self.alpha,
                        )
                        R[idx_is, j] = (
                            _minsum_f(R[idx_i, j - 1], L[idx_i, j], self.alpha)
                            + R[idx_is, j]
                        )

            num_iters = it

            # 早停检查
            for i in range(N):
                if self.frozen[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch_natural < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            if self.frozen[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
