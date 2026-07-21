"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from encoder import polar_encode, bit_reversal_permutation


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """
    BP 译码器。
    因子图有 n+1 层（层 0 到层 n），每层 N 个节点。
    层 0：信源比特端；层 n：信道接收端。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self.large = 1e6

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)

        L[n, :] = llr_ch[self.br]
        R[0, :] = 0.0
        R[0, self.frozen_bits.astype(bool)] = self.large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            for stage in range(n - 1, -1, -1):
                half = 2**stage
                block = 2 * half
                for base in range(0, N, block):
                    for k in range(base, base + half):
                        L[stage, k] = _f_min_sum(
                            L[stage + 1, k], L[stage + 1, k + half], self.alpha
                        )
                        L[stage, k + half] = _f_min_sum(
                            R[stage, k], L[stage + 1, k], self.alpha
                        ) + L[stage + 1, k + half]

            for stage in range(0, n):
                half = 2**stage
                block = 2 * half
                for base in range(0, N, block):
                    for k in range(base, base + half):
                        R[stage + 1, k] = _f_min_sum(
                            R[stage, k + half] + L[stage + 1, k + half],
                            R[stage, k],
                            self.alpha,
                        )
                        R[stage + 1, k + half] = _f_min_sum(
                            R[stage, k], L[stage + 1, k], self.alpha
                        ) + R[stage, k + half]

            u_hat = self._hard_decision(L, R)
            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
        else:
            u_hat = self._hard_decision(L, R)

        return u_hat, num_iters

    def _hard_decision(self, L, R):
        u_hat = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            total = L[0, i] + R[0, i]
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1
        return u_hat
