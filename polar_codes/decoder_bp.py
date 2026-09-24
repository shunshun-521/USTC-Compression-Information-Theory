"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from channel import hard_decision_llr


def _ms_f(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（因子图列 0=信源，列 n=信道）"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        alpha = self.alpha

        L = np.zeros((N, n + 1))
        R = np.zeros((N, n + 1))

        L[:, n] = llr_ch[self.br]

        for i in range(N):
            R[i, 0] = self.LARGE if self.frozen_bits[i] else 0.0

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            # 右到左更新 L 消息
            for s in range(n, 0, -1):
                span = 1 << (s - 1)
                for i in range(0, N, 2 * span):
                    for k in range(span):
                        idx_u = i + k
                        idx_v = i + k + span
                        L[idx_u, s - 1] = _ms_f(
                            R[idx_u, s] + L[idx_v, s], L[idx_u, s], alpha
                        )
                        L[idx_v, s - 1] = _ms_f(
                            R[idx_u, s], L[idx_u, s], alpha
                        ) + L[idx_v, s]

            # 左到右更新 R 消息
            for s in range(1, n + 1):
                span = 1 << (s - 1)
                for i in range(0, N, 2 * span):
                    for k in range(span):
                        idx_u = i + k
                        idx_v = i + k + span
                        R[idx_u, s] = _ms_f(
                            R[idx_v, s] + L[idx_v, s], R[idx_u, s - 1], alpha
                        )
                        R[idx_v, s] = _ms_f(
                            R[idx_u, s - 1], L[idx_u, s], alpha
                        ) + R[idx_v, s]

            u_hat = self._hard_decision(L, R)
            if self._check_early_stop(u_hat, llr_ch):
                num_iters = it
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters

    def _hard_decision(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits.astype(bool)] = 0
        return u_hat

    def _check_early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        x_hard = hard_decision_llr(llr_ch)
        return np.array_equal(x_hat, x_hard)
