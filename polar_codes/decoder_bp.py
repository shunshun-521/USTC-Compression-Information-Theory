"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from decoder_sc import f_operation, _decoder_domain_to_natural, _frozen_to_decoder_domain
from encoder import polar_encode, bit_reversal_permutation


class BPDecoder:
    """BP 译码器（因子图 min-sum，含早停）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = _frozen_to_decoder_domain(frozen_bits, N)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6
        self.br = bit_reversal_permutation(N)
        self.inv = np.argsort(self.br)

    def _f_min_sum(self, a, b):
        return self.alpha * f_operation(a, b)

    def _to_decoder_domain(self, u_natural):
        return np.asarray(u_natural, dtype=np.int32)[self.br]

    def decode(self, llr_ch):
        """主译码函数，返回自然序 u_hat 与迭代次数"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)

        L[:, 0] = llr_ch
        R[:, 0] = 0.0
        for idx in self.frozen_set:
            R[idx, 0] = self._large

        num_iters = 0
        u_prime = np.zeros(N, dtype=np.int32)

        for it in range(1, self.max_iter + 1):
            num_iters = it

            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        L[idx, j - 1] = self._f_min_sum(
                            R[idx, j] + L[idx + s, j], L[idx, j]
                        )
                        L[idx + s, j - 1] = self._f_min_sum(R[idx, j], L[idx, j]) + L[idx + s, j]

            for j in range(0, n):
                s = 1 << j
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        R[idx, j + 1] = self._f_min_sum(
                            R[idx + s, j] + L[idx + s, j + 1], R[idx, j]
                        )
                        R[idx + s, j + 1] = self._f_min_sum(R[idx, j], L[idx, j + 1]) + R[idx + s, j]

            for i in range(N):
                if i in self.frozen_set:
                    u_prime[i] = 0
                else:
                    u_prime[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

            u_natural = _decoder_domain_to_natural(u_prime, N)
            x_hat = polar_encode(u_natural)
            hard_ch = (llr_ch < 0).astype(np.int32)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            if i in self.frozen_set:
                u_prime[i] = 0
            else:
                u_prime[i] = 0 if (L[i, 0] + R[i, 0]) >= 0 else 1

        u_hat = _decoder_domain_to_natural(u_prime, N)
        return u_hat, num_iters
