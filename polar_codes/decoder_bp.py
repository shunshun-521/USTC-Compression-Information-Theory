"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode


def _bp_f(x, y, alpha):
    """BP 中使用的 min-sum f 运算。"""
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def decode(self, llr_ch):
        """主译码函数。"""
        N = self.N
        n = self.n
        alpha = self.alpha

        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        u_hat = np.zeros(N, dtype=int)
        num_iters = 0

        for it in range(self.max_iter):
            num_iters = it + 1

            # L 消息：从右向左（参考 Van den Brink 式 (3)）
            for stage in range(n):
                block = 2 ** stage
                stride = 2 * block
                for start in range(0, N, stride):
                    for offset in range(block):
                        i = start + offset
                        L[i, stage] = _bp_f(
                            L[i, stage + 1],
                            L[i + block, stage + 1] + R[i + block, stage],
                            alpha,
                        )
                        L[i + block, stage] = (
                            _bp_f(R[i, stage], L[i, stage + 1], alpha)
                            + L[i + block, stage + 1]
                        )

            # R 消息：从左向右
            for stage in range(n - 1, -1, -1):
                block = 2 ** stage
                stride = 2 * block
                for start in range(0, N, stride):
                    for offset in range(block):
                        i = start + offset
                        R[i, stage + 1] = _bp_f(
                            R[i, stage],
                            L[i + block, stage + 1] + R[i + block, stage],
                            alpha,
                        )
                        R[i + block, stage + 1] = (
                            _bp_f(R[i, stage], L[i, stage + 1], alpha)
                            + R[i + block, stage]
                        )

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
