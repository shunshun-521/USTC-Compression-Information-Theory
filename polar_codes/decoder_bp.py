"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import upper_llr


def ms_f(a, b, alpha=0.9375):
    """min-sum 修正的 f 运算"""
    result = upper_llr(a, b)
    if alpha != 1.0:
        return alpha * result
    return result


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
        self.LARGE = 1e6

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        alpha = self.alpha

        L_msg = np.zeros((N, n + 1), dtype=np.float64)
        R_msg = np.zeros((N, n + 1), dtype=np.float64)

        L_msg[:, 0] = llr_ch
        R_msg[:, n] = 0.0
        R_msg[self.frozen_idx, n] = self.LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    la = R_msg[i, j + 1] + L_msg[i + s, j + 1]
                    lb = L_msg[i, j]
                    L_msg[i, j + 1] = ms_f(la, lb, alpha)

                    la2 = R_msg[i, j + 1]
                    lb2 = L_msg[i, j]
                    L_msg[i + s, j + 1] = ms_f(la2, lb2, alpha) + L_msg[i + s, j + 1]

            for j in range(n - 1, -1, -1):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    la = R_msg[i + s, j] + L_msg[i + s, j + 1]
                    lb = R_msg[i, j + 1]
                    R_msg[i, j] = ms_f(la, lb, alpha)

                    la2 = R_msg[i, j + 1]
                    lb2 = L_msg[i, j + 1]
                    R_msg[i + s, j] = ms_f(la2, lb2, alpha) + R_msg[i + s, j + 1]

            total_llr = L_msg[:, n] + R_msg[:, n]
            u_hat = np.zeros(N, dtype=int)
            u_hat[self.info_idx] = (total_llr[self.info_idx] < 0).astype(int)

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        total_llr = L_msg[:, n] + R_msg[:, n]
        u_hat = np.zeros(N, dtype=int)
        u_hat[self.info_idx] = (total_llr[self.info_idx] < 0).astype(int)
        return u_hat, num_iters
