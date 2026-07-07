"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 列（列 0 到列 n），每列 N 个节点。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]
        self._build_offsets()

    def _build_offsets(self):
        """预计算每层段长。"""
        self.sp = [1 << i for i in range(self.n + 1)]

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        rev = bit_reversal_permutation(self.N)
        llr_ch = llr_ch[rev]
        n = self.n
        N = self.N
        LARGE = 1e6

        # L[i, j]: 从右到左消息; R[i, j]: 从左到右消息
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            # 从右到左更新 L
            for j in range(n, 0, -1):
                s = self.sp[j - 1]
                for block in range(0, N, 2 * s):
                    for i in range(block, block + s):
                        L[i, j - 1] = _f_min_sum(
                            R[i, j] + L[i + s, j + 1],
                            L[i, j + 1],
                            self.alpha,
                        )
                        L[i + s, j - 1] = (
                            _f_min_sum(R[i, j], L[i, j + 1], self.alpha)
                            + L[i + s, j + 1]
                        )

            # 从左到右更新 R
            for j in range(1, n + 1):
                s = self.sp[j - 1]
                for block in range(0, N, 2 * s):
                    for i in range(block, block + s):
                        R[i, j] = _f_min_sum(
                            R[i + s, j] + L[i + s, j + 1],
                            R[i, j - 1],
                            self.alpha,
                        )
                        R[i + s, j] = (
                            _f_min_sum(R[i, j - 1], L[i, j + 1], self.alpha)
                            + R[i + s, j]
                        )

            # 早停检查
            total = L[:, 0] + R[:, 0]
            u_hat = np.where(total >= 0, 0, 1).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.where(total >= 0, 0, 1).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
