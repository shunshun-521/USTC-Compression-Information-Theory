"""
极化码 BP（置信传播）译码器
基于因子图，min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_indices = set(np.where(self.frozen_bits == 1)[0])
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _f_minsum(self, a, b):
        return self.alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        # L[i][j]: 从右到左消息，R[i][j]: 从左到右消息
        # j = 0..n, 每列 N 个节点
        L_msg = np.zeros((N, n + 1), dtype=np.float64)
        R_msg = np.zeros((N, n + 1), dtype=np.float64)

        L_msg[:, n] = llr_ch.copy()
        R_msg[:, 0] = 0.0
        for idx in self.frozen_indices:
            R_msg[idx, 0] = self.LARGE

        num_iters = self.max_iter

        for it in range(self.max_iter):
            # 从右到左更新 L 消息 (j = n 到 1)
            for j in range(n, 0, -1):
                s = 2 ** (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_u = i + k
                        idx_v = i + k + s
                        R_u = R_msg[idx_u, j - 1]
                        L_v = L_msg[idx_v, j]
                        L_u = L_msg[idx_u, j]
                        R_v = R_msg[idx_v, j - 1]
                        L_v_next = L_msg[idx_v, j]

                        L_msg[idx_u, j - 1] = self._f_minsum(R_u + L_v, L_u)
                        L_msg[idx_v, j - 1] = self._f_minsum(R_u, L_u) + L_v_next

            # 从左到右更新 R 消息 (j = 0 到 n-1)
            for j in range(0, n):
                s = 2 ** (j + 1)
                half = s // 2
                for i in range(0, N, s):
                    for k in range(half):
                        idx_u = i + k
                        idx_v = i + k + half
                        R_v = R_msg[idx_v, j]
                        L_v = L_msg[idx_v, j + 1]
                        R_u = R_msg[idx_u, j]
                        L_u = L_msg[idx_u, j + 1]
                        R_v_next = R_msg[idx_v, j + 1]

                        R_msg[idx_u, j + 1] = self._f_minsum(R_v + L_v, R_u)
                        R_msg[idx_v, j + 1] = self._f_minsum(R_u, L_u) + R_v_next

            # 早停检查
            total_llr = L_msg[:, 0] + R_msg[:, 0]
            u_hat = np.zeros(N, dtype=int)
            for i in range(N):
                if i in self.frozen_indices:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total_llr[i] >= 0 else 1

            x_hat = polar_encode(u_hat)
            x_hard = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, x_hard):
                num_iters = it + 1
                break

        total_llr = L_msg[:, 0] + R_msg[:, 0]
        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            if i in self.frozen_indices:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total_llr[i] >= 0 else 1

        return u_hat, num_iters
