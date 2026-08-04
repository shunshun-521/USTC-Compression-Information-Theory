"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, _butterfly_encode


def _f_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（因子图 min-sum，含早停）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self._large = 1e6

        rev = np.zeros(N, dtype=np.int64)
        for i in range(N):
            r = 0
            v = i
            for _ in range(self.n):
                r = (r << 1) | (v & 1)
                v >>= 1
            rev[i] = r
        self._rev = rev

    def _prepare_llr(self, llr_ch):
        llr = np.asarray(llr_ch, dtype=np.float64).copy()
        return llr[self._rev]

    def decode(self, llr_ch):
        llr_ch = self._prepare_llr(llr_ch)
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self._large

        num_iters = 0
        u_hat = np.zeros(N, dtype=np.int64)

        for it in range(self.max_iter):
            num_iters = it + 1

            # 右到左更新 L
            for j in range(n, 0, -1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        L[idx, j - 1] = _f_min_sum(
                            R[idx, j - 1] + L[idx + s, j],
                            L[idx, j], self.alpha
                        )
                        L[idx + s, j - 1] = _f_min_sum(
                            R[idx, j - 1], L[idx, j], self.alpha
                        ) + L[idx + s, j]

            # 左到右更新 R
            R_new = R.copy()
            for j in range(1, n + 1):
                s = 1 << (j - 1)
                for i in range(0, N, 2 * s):
                    for k in range(s):
                        idx = i + k
                        R_new[idx, j] = _f_min_sum(
                            R[idx + s, j - 1] + L[idx + s, j],
                            R[idx, j - 1], self.alpha
                        )
                        R_new[idx + s, j] = _f_min_sum(
                            R[idx, j - 1], L[idx, j], self.alpha
                        ) + R[idx + s, j - 1]
            R = R_new

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if total >= 0 else 1

            x_hat_bf = _butterfly_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(np.int64)
            if np.array_equal(x_hat_bf, hard_ch):
                break

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if total >= 0 else 1

        return u_hat, num_iters
