"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import _sign_pm


def checknode(a, b, alpha=0.9375):
    """min-sum 校验节点运算（向量化）。"""
    return alpha * _sign_pm(a) * _sign_pm(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)
        self.LARGE = 1e6
        self._masks = []
        self._add_ks = []
        for i in range(self.n):
            add_k = N // (2 ** (i + 1))
            mask = np.array([j for j in range(N) if j % (2 * add_k) < add_k], dtype=int)
            self._masks.append(mask)
            self._add_ks.append(add_k)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, num_iters)
        """
        N, n = self.N, self.n
        llr = llr_ch[self.br].astype(np.float64)

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(self.max_iter):
            num_iters = it + 1

            for i in range(n):
                add_k = self._add_ks[i]
                mask = self._masks[i]
                R[mask, i + 1] = checknode(
                    R[mask, i], L[mask + add_k, i + 1] + R[mask + add_k, i], self.alpha
                )
                R[mask + add_k, i + 1] = checknode(
                    R[mask, i], L[mask, i + 1], self.alpha
                ) + R[mask + add_k, i]

            for i in range(n - 1, -1, -1):
                add_k = self._add_ks[i]
                mask = self._masks[i]
                L[mask, i] = checknode(
                    L[mask, i + 1], L[mask + add_k, i + 1] + R[mask + add_k, i], self.alpha
                )
                L[mask + add_k, i] = checknode(
                    R[mask, i], L[mask, i + 1], self.alpha
                ) + L[mask + add_k, i + 1]

            total = L[:, 0] + R[:, 0]
            u_hat = np.where(self.frozen_bits, 0, (total < 0).astype(int))

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        total = L[:, 0] + R[:, 0]
        u_hat = np.where(self.frozen_bits, 0, (total < 0).astype(int))
        return u_hat, num_iters
