"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import bit_reversal_permutation, polar_encode


def _min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


def _build_mask_dict(N):
    m = int(np.log2(N))
    masks = []
    for i in range(m):
        i_back = m - i - 1
        add_k = N // (2 ** (i_back + 1))
        block = 2 * add_k
        mask = []
        for start in range(0, N, block):
            for k in range(add_k):
                mask.append(start + k)
        masks.append(np.array(mask, dtype=np.int64))
    return masks


class BPDecoder:
    """BP 译码器（min-sum + 早停）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.m = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.clip = 1e6
        self.mask_dict = _build_mask_dict(N)

    def _update_left(self, R, L):
        for i in range(self.m - 1, -1, -1):
            i_back = self.m - i - 1
            add_k = self.N // (2 ** (i_back + 1))
            mask = self.mask_dict[i]
            if len(mask) == 0:
                continue
            L[i, mask] = _min_sum(
                L[i + 1, mask],
                L[i + 1, mask + add_k] + R[i, mask + add_k],
                self.alpha,
            )
            L[i, mask + add_k] = _min_sum(
                R[i, mask], L[i + 1, mask], self.alpha
            ) + L[i + 1, mask + add_k]
        np.clip(L, -self.clip, self.clip, out=L)

    def _update_right(self, R, L):
        for i in range(self.m):
            i_back = self.m - i - 1
            add_k = self.N // (2 ** (i_back + 1))
            mask = self.mask_dict[i]
            if len(mask) == 0:
                continue
            R[i + 1, mask] = _min_sum(
                R[i, mask],
                L[i + 1, mask + add_k] + R[i, mask + add_k],
                self.alpha,
            )
            R[i + 1, mask + add_k] = _min_sum(
                R[i, mask], L[i + 1, mask], self.alpha
            ) + R[i, mask + add_k]
        np.clip(R, -self.clip, self.clip, out=R)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        rev = bit_reversal_permutation(self.N)
        llr_int = llr_ch[rev]
        R = np.zeros((self.m + 1, self.N), dtype=np.float64)
        L = np.zeros((self.m + 1, self.N), dtype=np.float64)
        L[self.m, :] = llr_int
        R[0, self.frozen_bits] = self.clip

        hard_ch = (llr_int < 0).astype(int)

        for it in range(1, self.max_iter + 1):
            self._update_left(R, L)
            self._update_right(R, L)

            u_soft = L[0, :] + R[0, :]
            u_hat = (u_soft < 0).astype(int)
            u_hat[self.frozen_bits] = 0
            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat[rev], hard_ch):
                return u_hat, it

        u_soft = L[0, :] + R[0, :]
        u_hat = (u_soft < 0).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, self.max_iter
