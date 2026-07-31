"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode


def _f_min_sum(x, y, alpha):
    """scaled min-sum f 运算"""
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """
    BP 译码器。
    因子图 m+1 列（0=信源端，m=信道端），每列 N 个节点。
    """

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.m = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]

    def _decode_u_from_messages(self, L, R):
        """从信道端总 LLR 硬判决码字，再经极化逆变换得到 u"""
        x_hat = ((L[:, self.m] + R[:, self.m]) < 0).astype(int)
        u_hat = polar_encode(x_hat)
        for i in self.frozen_idx:
            u_hat[i] = 0
        return u_hat

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        m = self.m
        N = self.N
        alpha = self.alpha

        L = np.zeros((N, m + 1), dtype=np.float64)
        R = np.zeros((N, m + 1), dtype=np.float64)

        L[:, m] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter

        for it in range(self.max_iter):
            L_prev = L.copy()

            for j in range(1, m + 1):
                step = 1 << (m - j)
                col = j - 1
                for i in range(0, N, 2 * step):
                    ib = i + step
                    L[ib, col] = L_prev[ib, col + 1] + _f_min_sum(
                        L_prev[i, col + 1], R[i, col], alpha
                    )
                    L[i, col] = _f_min_sum(
                        L_prev[i, col + 1],
                        L_prev[ib, col + 1] + R[ib, col],
                        alpha,
                    )

            for j in range(1, m + 1):
                step = 1 << (m - j)
                col = j - 1
                for i in range(0, N, 2 * step):
                    ib = i + step
                    R[i, col + 1] = _f_min_sum(
                        R[i, col], L_prev[ib, col + 1] + R[ib, col], alpha
                    )
                    R[ib, col + 1] = R[ib, col] + _f_min_sum(
                        L_prev[i, col + 1], R[i, col], alpha
                    )

            u_hat = self._decode_u_from_messages(L, R)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it + 1
                break

        u_hat = self._decode_u_from_messages(L, R)
        return u_hat, num_iters
