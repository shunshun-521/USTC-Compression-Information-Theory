"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation


def _min_sum_f(a, b, alpha=1.0):
    sa = np.sign(a)
    sb = np.sign(b)
    if sa == 0:
        sa = 1
    if sb == 0:
        sb = 1
    return alpha * sa * sb * min(abs(a), abs(b))


class BPDecoder:
    """BP 译码器（分层因子图 min-sum BP）。"""

    LARGE = 1e7

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.rev = bit_reversal_permutation(N)

    def _f_ms(self, a, b):
        return _min_sum_f(a, b, self.alpha)

    def _pe_update(self, L1, L2, R1, R2):
        """单个处理单元（PE）的 min-sum BP 更新。"""
        T = self._f_ms(R1 + L2, L1)
        L1_new = T
        L2_new = self._f_ms(R1, L1) + L2
        R2_new = self._f_ms(R2, L1) + R1
        R1_new = self._f_ms(R2 + L2_new, R1)
        return L1_new, L2_new, R1_new, R2_new

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        llr_br = llr_ch[self.rev]

        # stage 0: 信源端，stage n: 信道端
        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_br

        for i in range(N):
            if self.frozen_bits[i] == 1:
                R[0, i] = self.LARGE
            else:
                R[0, i] = 0.0

        num_iters = 0
        for it in range(self.max_iter):
            num_iters = it + 1

            # 左向传播：更新 L（从 stage n 到 1）
            for s in range(n, 0, -1):
                step = 2 ** (n - s)
                for j in range(0, N, 2 * step):
                    for i in range(j, j + step):
                        L[s - 1, i], L[s - 1, i + step], R[s - 1, i + step], R[s - 1, i] = (
                            self._pe_update(
                                L[s, i], L[s, i + step], R[s - 1, i], R[s - 1, i + step]
                            )
                        )

            # 右向传播：更新 R（从 stage 0 到 n-1）
            for s in range(0, n):
                step = 2 ** (n - s - 1)
                for j in range(0, N, 2 * step):
                    for i in range(j, j + step):
                        L[s + 1, i], L[s + 1, i + step], R[s + 1, i + step], R[s + 1, i] = (
                            self._pe_update(
                                L[s + 1, i], L[s + 1, i + step], R[s, i], R[s, i + step]
                            )
                        )

            # 冻结位约束
            for i in range(N):
                if self.frozen_bits[i] == 1:
                    R[0, i] = self.LARGE

            u_hat = self._hard_decision(L, R)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        u_hat = self._hard_decision(L, R)
        return u_hat, num_iters

    def _hard_decision(self, L, R):
        u_hat = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            if self.frozen_bits[i] == 1:
                u_hat[i] = 0
            else:
                total = L[0, i] + R[0, i]
                u_hat[i] = 0 if total >= 0 else 1
        return u_hat
