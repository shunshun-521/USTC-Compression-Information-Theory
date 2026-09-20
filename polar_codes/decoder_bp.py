"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, map_decoded_to_u_domain, transform_frozen_bits


class BPDecoder:
    """BP 译码器（u' 域因子图，min-sum 近似）"""

    LARGE = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375, u_domain=True):
        self.N = N
        self.n = int(np.log2(N))
        self.max_iter = max_iter
        self.alpha = alpha
        self.u_domain = u_domain

        frozen_bits = np.asarray(frozen_bits)
        info_idx = np.where(frozen_bits == 0)[0]
        if u_domain:
            self.frozen_bits = transform_frozen_bits(frozen_bits, info_idx, N)
        else:
            self.frozen_bits = frozen_bits.astype(int)

    @staticmethod
    def _f_ms(x, y, alpha):
        return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 u 域 u_hat 与实际迭代次数。
        """
        from decoder_sc import _prepare_llr

        llr_ch = _prepare_llr(llr_ch, self.N)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits.astype(bool), 0] = self.LARGE

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for j in range(n, 0, -1):
                step = 1 << (j - 1)
                for i in range(0, N, step << 1):
                    for t in range(step):
                        idx = i + t
                        La = R[idx, j - 1] + L[idx + step, j]
                        Lb = L[idx, j]
                        L[idx, j - 1] = self._f_ms(La, Lb, self.alpha)
                        L[idx + step, j - 1] = self._f_ms(R[idx, j - 1], Lb, self.alpha) + L[idx + step, j]

            for j in range(1, n + 1):
                step = 1 << (j - 1)
                for i in range(0, N, step << 1):
                    for t in range(step):
                        idx = i + t
                        Ra = R[idx + step, j] + L[idx + step, j]
                        Rb = R[idx, j - 1]
                        R[idx, j] = self._f_ms(Ra, Lb := L[idx, j], self.alpha)
                        R[idx + step, j] = self._f_ms(Rb, Lb, self.alpha) + R[idx + step, j - 1]

            total = L[:, 0] + R[:, 0]
            u_prime = np.zeros(N, dtype=int)
            u_prime[self.frozen_bits.astype(bool)] = 0
            info_mask = self.frozen_bits == 0
            u_prime[info_mask] = (total[info_mask] < 0).astype(int)

            x_hat = polar_encode(map_decoded_to_u_domain(u_prime, N)) if self.u_domain else u_prime
            hard_x = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_x):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_prime = np.zeros(N, dtype=int)
        u_prime[self.frozen_bits.astype(bool)] = 0
        info_mask = self.frozen_bits == 0
        u_prime[info_mask] = (total[info_mask] < 0).astype(int)

        if self.u_domain:
            u_hat = map_decoded_to_u_domain(u_prime, N)
        else:
            u_hat = u_prime
        return u_hat, num_iters
