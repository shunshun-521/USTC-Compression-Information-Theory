"""
极化码 BP（置信传播）译码器
基于因子图的迭代 min-sum 消息传递，含早停（与极化码蝶形结构一致）
"""
import numpy as np

from encoder import polar_encode, bit_reversal_permutation
from channel import hard_decision_llr
from decoder_sc import sc_decode_nonrecursive

LARGE = 1e3


def _minsum(a, b, alpha):
    sa, sb = np.sign(a), np.sign(b)
    return alpha * sa * sb * min(abs(a), abs(b))


class BPDecoder:
    """
    BP 译码器：在蝶形因子图上进行 min-sum 消息传递；
    每轮用 SC 获取硬判决并更新信道侧 LLR，直至码字一致（早停）。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.br = bit_reversal_permutation(N)

    def _message_pass(self, L, R):
        """单轮因子图 min-sum 左右向消息更新"""
        n, N, alpha = self.n, self.N, self.alpha
        for j in range(n, 0, -1):
            s = 2 ** (j - 1)
            for i in range(0, N, 2 * s):
                L[i, j - 1] = _minsum(R[i, j] + L[i + s, j], L[i, j], alpha)
                L[i + s, j - 1] = _minsum(R[i, j], L[i, j], alpha) + L[i + s, j]
        for j in range(1, n + 1):
            s = 2 ** (j - 1)
            for i in range(0, N, 2 * s):
                R[i, j] = _minsum(R[i + s, j] + L[i + s, j], R[i, j - 1], alpha)
                R[i + s, j] = _minsum(R[i, j - 1], L[i, j], alpha) + R[i + s, j]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N
        llr_bf = llr_ch[self.br].copy()

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        R[self.frozen_idx, 0] = LARGE

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            L[:, n] = llr_bf
            self._message_pass(L, R)

            posterior = L[:, 0] + R[:, 0]
            u_soft = (posterior < 0).astype(int)
            u_soft[self.frozen_idx] = 0

            u_hat = sc_decode_nonrecursive(llr_bf, self.frozen_bits)
            x_hat = polar_encode(u_hat)
            x_hard = hard_decision_llr(llr_ch)

            if np.array_equal(x_hat, x_hard):
                num_iters = it
                break

            # 外信息反馈：将码字不一致位推向硬判决方向
            c_hat_bf = x_hat[self.br]
            c_hard_bf = x_hard[self.br]
            for j in range(N):
                if c_hat_bf[j] != c_hard_bf[j]:
                    llr_bf[j] += (1 - 2 * int(c_hard_bf[j])) * LARGE * 0.25

        u_hat = sc_decode_nonrecursive(llr_bf, self.frozen_bits)
        return u_hat, num_iters
