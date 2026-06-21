"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode


def _ms_f(a, b, alpha):
    """min-sum 近似 f 函数。"""
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha

    def _hard_codeword(self, llr_ch):
        """对信道 LLR 做硬判决得到码字。"""
        return (llr_ch < 0).astype(int)

    def _decide(self, L_msg, R_msg):
        """最左列判决。"""
        u_hat = np.zeros(self.N, dtype=int)
        total = L_msg[:, 0] + R_msg[:, 0]
        u_hat[total < 0] = 1
        u_hat[self.frozen_bits] = 0
        return u_hat

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

        L_msg = np.zeros((N, n + 1), dtype=np.float64)
        R_msg = np.zeros((N, n + 1), dtype=np.float64)

        L_msg[:, n] = llr_ch
        R_msg[:, 0] = 0.0
        R_msg[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_u = i + k
                        idx_l = i + k + s
                        L_msg[idx_u, j - 1] = _ms_f(
                            R_msg[idx_u, j - 1] + L_msg[idx_l, j],
                            L_msg[idx_u, j],
                            self.alpha,
                        )
                        L_msg[idx_l, j - 1] = _ms_f(
                            R_msg[idx_u, j - 1],
                            L_msg[idx_u, j],
                            self.alpha,
                        ) + L_msg[idx_l, j]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx_u = i + k
                        idx_l = i + k + s
                        R_msg[idx_u, j + 1] = _ms_f(
                            R_msg[idx_l, j] + L_msg[idx_l, j + 1],
                            R_msg[idx_u, j],
                            self.alpha,
                        )
                        R_msg[idx_l, j + 1] = _ms_f(
                            R_msg[idx_u, j],
                            L_msg[idx_l, j + 1],
                            self.alpha,
                        ) + R_msg[idx_l, j]

            u_hat = self._decide(L_msg, R_msg)
            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, self._hard_codeword(llr_ch)):
                break

        u_hat = self._decide(L_msg, R_msg)
        return u_hat, num_iters
