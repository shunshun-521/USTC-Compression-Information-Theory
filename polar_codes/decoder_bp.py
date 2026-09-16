"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import math
import numpy as np

from decoder_sc import prepare_channel_llr
from encoder import polar_encode


def _checknode(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.clip = 1e6

    def _update_left(self, L, R):
        m = self.n
        N = self.N
        perm = list(range(m))
        for i in reversed(perm):
            i_back = m - i - 1
            add_k = N // (2 ** (i_back + 1))
            for base in range(0, N, 2 * add_k):
                for off in range(add_k):
                    idx = base + off
                    s = idx + add_k
                    L[idx, i] = _checknode(
                        L[idx, i + 1],
                        L[s, i + 1] + R[s, i],
                        self.alpha,
                    )
                    L[s, i] = _checknode(R[idx, i], L[idx, i + 1], self.alpha) + L[s, i + 1]
        return np.clip(L, -self.clip, self.clip)

    def _update_right(self, L, R):
        m = self.n
        N = self.N
        perm = list(range(m))
        for i in perm:
            i_back = m - i - 1
            add_k = N // (2 ** (i_back + 1))
            for base in range(0, N, 2 * add_k):
                for off in range(add_k):
                    idx = base + off
                    s = idx + add_k
                    R[idx, i + 1] = _checknode(
                        R[idx, i],
                        L[s, i + 1] + R[s, i],
                        self.alpha,
                    )
                    R[s, i + 1] = _checknode(R[idx, i], L[idx, i + 1], self.alpha) + R[s, i]
        return np.clip(R, -self.clip, self.clip)

    def _hard_codeword(self, llr_ch):
        return (llr_ch < 0).astype(int)

    def decode(self, llr_ch):
        llr = prepare_channel_llr(llr_ch)
        m = self.n
        N = self.N

        L = np.zeros((N, m + 1), dtype=np.float64)
        R = np.zeros((N, m + 1), dtype=np.float64)
        L[:, m] = llr
        R[:, 0] = 0.0
        R[self.frozen_bits, 0] = self.clip

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            L = self._update_left(L, R)
            R = self._update_right(L, R)

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, self._hard_codeword(llr)):
                num_iters = it
                break

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
