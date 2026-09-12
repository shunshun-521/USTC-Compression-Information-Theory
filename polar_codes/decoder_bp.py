"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode


def _checknode(a, b, alpha):
    """min-sum 校验节点运算"""
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（含循环置换以改善收敛）"""

    CLIP = 1e7

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.m = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.mask_dict = self._build_masks()

    def _build_masks(self):
        masks = {}
        for i in range(self.m):
            i_back = self.m - i - 1
            add_k = self.N // (2 ** (i_back + 1))
            masks[i] = np.arange(0, self.N, 2 * add_k, dtype=int)
        return masks

    def _update_left(self, L, R, perm):
        for i in perm[::-1]:
            i_back = self.m - i - 1
            add_k = self.N // (2 ** (i_back + 1))
            mask = self.mask_dict[i]
            m2 = mask + add_k
            L[mask, i] = _checknode(
                L[mask, i + 1], L[m2, i + 1] + R[m2, i], self.alpha
            )
            L[m2, i] = _checknode(
                R[mask, i], L[mask, i + 1], self.alpha
            ) + L[m2, i + 1]
            np.clip(L[:, i], -self.CLIP, self.CLIP, out=L[:, i])
        return L

    def _update_right(self, R, L, perm):
        for i in perm:
            i_back = self.m - i - 1
            add_k = self.N // (2 ** (i_back + 1))
            mask = self.mask_dict[i]
            m2 = mask + add_k
            R[mask, i + 1] = _checknode(
                R[mask, i], L[m2, i + 1] + R[m2, i], self.alpha
            )
            R[m2, i + 1] = _checknode(
                R[mask, i], L[mask, i + 1], self.alpha
            ) + R[m2, i]
            np.clip(R[:, i + 1], -self.CLIP, self.CLIP, out=R[:, i + 1])
        return R

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        m = self.m
        R = np.zeros((self.N, m + 1), dtype=np.float64)
        L = np.zeros((self.N, m + 1), dtype=np.float64)

        L[:, m] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.CLIP

        num_iters = 0
        u_hat = np.zeros(self.N, dtype=int)
        base_perm = np.arange(m)

        for iteration in range(self.max_iter):
            num_iters = iteration + 1
            perm = np.roll(base_perm, iteration % m)
            L = self._update_left(L, R, perm)
            R = self._update_right(R, L, perm)

            for i in range(self.N):
                u_hat[i] = 0 if self.frozen_bits[i] or (L[i, 0] + R[i, 0]) >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_decision = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_decision):
                break

        for i in range(self.N):
            u_hat[i] = 0 if self.frozen_bits[i] or (L[i, 0] + R[i, 0]) >= 0 else 1

        return u_hat, num_iters
