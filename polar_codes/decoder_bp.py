"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode
from decoder_sc import f_operation_minsum as f_ms


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.LARGE = 1e6

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for lam in range(n - 1, -1, -1):
                step = 1 << lam
                for i in range(0, N, step * 2):
                    for j in range(step):
                        idx = i + j
                        s = step
                        r_val = R[idx, lam]
                        l_upper = L[idx + s, lam + 1]
                        l_self = L[idx, lam + 1]
                        l_sib = L[idx + s, lam + 1]

                        L[idx, lam] = self.alpha * f_ms(r_val + l_upper, l_self)
                        L[idx + s, lam] = f_ms(r_val, l_self) + l_sib

            for lam in range(1, n + 1):
                step = 1 << (lam - 1)
                for i in range(0, N, step * 2):
                    for j in range(step):
                        idx = i + j
                        s = step
                        r_sib = R[idx + s, lam]
                        l_sib = L[idx + s, lam]
                        r_prev = R[idx, lam - 1]

                        R[idx, lam] = self.alpha * f_ms(r_sib + l_sib, r_prev)
                        R[idx + s, lam] = f_ms(r_prev, L[idx, lam]) + r_sib

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            u_hat[i] = 0 if total >= 0 else 1
        u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
