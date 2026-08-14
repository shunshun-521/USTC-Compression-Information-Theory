"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
import math

from encoder import polar_encode, bit_reversal_permutation


def _bp_f(x, y, alpha):
    """min-sum f 运算（带修正因子 alpha）。"""
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.rev = bit_reversal_permutation(N)
        self.large = 1e6

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        alpha = self.alpha

        rev = self.rev
        llr_work = llr_ch[rev]

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_work
        R[:, 0] = 0.0
        R[self.frozen_bits == 1, 0] = self.large

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            # 从右到左更新 L 消息
            for j in range(n, 0, -1):
                s = 2 ** (j - 1)
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        idx_u = i + t
                        idx_v = i + t + s
                        L[idx_u, j - 1] = _bp_f(
                            R[idx_u, j] + L[idx_v, j],
                            L[idx_u, j],
                            alpha,
                        )
                        L[idx_v, j - 1] = _bp_f(
                            R[idx_u, j],
                            L[idx_u, j],
                            alpha,
                        ) + L[idx_v, j]

            # 从左到右更新 R 消息
            for j in range(1, n + 1):
                s = 2 ** (j - 1)
                for i in range(0, N, 2 * s):
                    for t in range(s):
                        idx_u = i + t
                        idx_v = i + t + s
                        R[idx_u, j] = _bp_f(
                            R[idx_v, j] + L[idx_v, j],
                            R[idx_u, j - 1],
                            alpha,
                        )
                        R[idx_v, j] = _bp_f(
                            R[idx_u, j - 1],
                            L[idx_v, j],
                            alpha,
                        ) + R[idx_v, j]

            # 早停检查
            u_hat = self._hard_decision(L, R)
            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters

    def _hard_decision(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits == 1] = 0
        return u_hat
