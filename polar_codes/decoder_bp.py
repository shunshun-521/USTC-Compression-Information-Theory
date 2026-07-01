"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np
from decoder_sc import f_operation
from encoder import polar_encode


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        fb = np.asarray(frozen_bits)
        if fb.dtype == bool:
            self.frozen_mask = fb
            self.frozen_idx = np.where(fb)[0]
            self.info_idx = np.where(~fb)[0]
        else:
            fi = fb.astype(int)
            self.frozen_mask = fi != 0
            self.frozen_idx = np.where(fi)[0]
            self.info_idx = np.where(fi == 0)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6

    def _g(self, x, y):
        return self.alpha * f_operation(x, y)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)

        L[n, :] = llr_ch
        R[0, self.info_idx] = 0.0
        R[0, self.frozen_idx] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            L_old = L.copy()
            for s in range(n - 1, -1, -1):
                block = 1 << s
                for j in range(0, N, 2 * block):
                    for k in range(block):
                        a, b = j + k, j + k + block
                        L[s, a] = self._g(
                            L_old[s + 1, a], L_old[s + 1, b] + R[s, b]
                        )
                        L[s, b] = self._g(L_old[s + 1, a], R[s, a]) + L_old[s + 1, b]

            R_old = R.copy()
            for s in range(0, n):
                block = 1 << s
                for j in range(0, N, 2 * block):
                    for k in range(block):
                        a, b = j + k, j + k + block
                        R[s + 1, a] = self._g(
                            R_old[s, a], L_old[s + 1, b] + R_old[s, b]
                        )
                        R[s + 1, b] = (
                            self._g(L_old[s + 1, a], R_old[s, a]) + R_old[s, b]
                        )

            for i in range(N):
                if self.frozen_mask[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if (L[0, i] + R[0, i]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break
            num_iters = it

        for i in range(N):
            if self.frozen_mask[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if (L[0, i] + R[0, i]) >= 0 else 1

        return u_hat, num_iters
