"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode


def _f_minsum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（Arikan 因子图，min-sum）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.max_iter = max_iter
        self.alpha = alpha
        fb = np.asarray(frozen_bits)
        if fb.dtype == bool:
            self.frozen = fb
        elif set(np.unique(fb)).issubset({0, 1}):
            self.frozen = fb.astype(bool)
        else:
            frozen_set = set(int(x) for x in fb)
            self.frozen = np.array([i in frozen_set for i in range(N)], dtype=bool)
        self.large = 1e6

    def decode(self, llr_ch):
        """
        BP 译码。

        返回：
            u_hat: 估计源序列
            num_iters: 实际迭代次数
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        alpha = self.alpha

        L_msg = np.zeros((N, n + 1), dtype=np.float64)
        R_msg = np.zeros((N, n + 1), dtype=np.float64)
        L_msg[:, n] = llr_ch
        R_msg[:, 0] = 0.0
        R_msg[self.frozen, 0] = self.large

        for it in range(1, self.max_iter + 1):
            # 右到左更新 L
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    Li = i
                    Li2 = i + step
                    f_in = R_msg[Li, j] + L_msg[Li2, j]
                    L_msg[Li, j - 1] = _f_minsum(f_in, L_msg[Li, j], alpha)
                    L_msg[Li2, j - 1] = (
                        _f_minsum(R_msg[Li, j], L_msg[Li, j], alpha) + L_msg[Li2, j]
                    )

            # 左到右更新 R
            for j in range(1, n + 1):
                step = 1 << (j - 1)
                for i in range(0, N, 2 * step):
                    Li = i
                    Li2 = i + step
                    R_msg[Li, j] = _f_minsum(
                        R_msg[Li2, j] + L_msg[Li2, j], R_msg[Li, j - 1], alpha
                    )
                    R_msg[Li2, j] = (
                        _f_minsum(R_msg[Li, j - 1], L_msg[Li, j], alpha) + R_msg[Li2, j]
                    )

            total = L_msg[:, 0] + R_msg[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                return u_hat, it

        total = L_msg[:, 0] + R_msg[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen] = 0
        return u_hat, self.max_iter
