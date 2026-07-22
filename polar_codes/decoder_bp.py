"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from channel import hard_decision_llr


def _g_bp_vec(a, b, alpha):
    """向量化的 min-sum BP 运算"""
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """
    BP 译码器。
    因子图有 m+1 个阶段（0 到 m），阶段 m 为信道接收端。
    """

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.m = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        rev = bit_reversal_permutation(self.N)
        llr_ch = llr_ch[rev]

        m = self.m
        N = self.N
        alpha = self.alpha

        L = np.zeros((m + 1, N), dtype=np.float64)
        R = np.zeros((m + 1, N), dtype=np.float64)

        L[m, :] = llr_ch
        R[0, :] = 0.0
        R[0, self.frozen_idx] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for i in range(m - 1, -1, -1):
                stride = 1 << i
                for j in range(0, N, 2 * stride):
                    sl = slice(j, j + stride)
                    sr = slice(j + stride, j + 2 * stride)
                    L[i, sl] = _g_bp_vec(
                        L[i + 1, sl], L[i + 1, sr] + R[i, sr], alpha
                    )
                    L[i, sr] = _g_bp_vec(L[i + 1, sl], R[i, sl], alpha) + L[i + 1, sr]

            for i in range(0, m):
                stride = 1 << i
                for j in range(0, N, 2 * stride):
                    sl = slice(j, j + stride)
                    sr = slice(j + stride, j + 2 * stride)
                    R[i + 1, sl] = _g_bp_vec(
                        R[i, sl], L[i + 1, sr] + R[i, sr], alpha
                    )
                    R[i + 1, sr] = _g_bp_vec(L[i + 1, sl], R[i, sl], alpha) + R[i, sr]

            u_hat = (L[0, :] < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)
            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break
            num_iters = it

        u_hat = (L[0, :] < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
