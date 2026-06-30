"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import polar_encode


def _cn_min_sum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器。"""

    LARGE = 1e6
    CLIP = 1e6

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.m = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self._masks = [
            np.arange(0, N, 2 * (1 << stage))
            for stage in range(self.m)
        ]

    def _update_left(self, L, R):
        for stage in range(self.m):
            mask = self._masks[stage]
            step = 1 << stage
            L[mask, stage] = _cn_min_sum(
                L[mask, stage + 1],
                L[mask + step, stage + 1] + R[mask + step, stage],
                self.alpha,
            )
            L[mask + step, stage] = np.clip(
                _cn_min_sum(R[mask, stage], L[mask, stage + 1], self.alpha)
                + L[mask + step, stage + 1],
                -self.CLIP, self.CLIP,
            )
        return np.clip(L, -self.CLIP, self.CLIP)

    def _update_right(self, R, L):
        for stage in range(self.m - 1, -1, -1):
            mask = self._masks[stage]
            step = 1 << stage
            R[mask, stage + 1] = _cn_min_sum(
                R[mask + step, stage],
                L[mask + step, stage + 1] + R[mask, stage],
                self.alpha,
            )
            R[mask + step, stage + 1] = np.clip(
                _cn_min_sum(R[mask, stage], L[mask, stage + 1], self.alpha)
                + R[mask + step, stage],
                -self.CLIP, self.CLIP,
            )
        return np.clip(R, -self.CLIP, self.CLIP)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        m = self.m
        N = self.N

        L = np.zeros((N, m + 1), dtype=np.float64)
        R = np.zeros((N, m + 1), dtype=np.float64)

        L[:, m] = llr_ch
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.LARGE

        num_iters = 0
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            L = self._update_left(L, R)
            R = self._update_right(R, L)

            for i in range(N):
                total = L[i, 0] + R[i, 0]
                u_hat[i] = 0 if total >= 0 else 1
            u_hat[self.frozen_idx] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        for i in range(N):
            total = L[i, 0] + R[i, 0]
            u_hat[i] = 0 if total >= 0 else 1
        u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
