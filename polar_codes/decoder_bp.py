"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _ms_f(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（min-sum，含早停）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_indices = set(np.where(self.frozen_bits > 0)[0])
        self.max_iter = max_iter
        self.alpha = alpha
        self.large = 1e6

    def _update_left(self, L, R):
        """从右向左更新 L 消息。"""
        n = self.n
        N = self.N
        alpha = self.alpha
        for stage in range(n - 1, -1, -1):
            block = 1 << (stage + 1)
            half = block // 2
            for start in range(0, N, block):
                for offset in range(half):
                    top = start + offset
                    bot = start + offset + half
                    L[top, stage] = _ms_f(
                        L[top, stage + 1], L[bot, stage + 1] + R[bot, stage], alpha
                    )
                    L[bot, stage] = (
                        _ms_f(R[top, stage], L[top, stage + 1], alpha) + L[bot, stage + 1]
                    )

    def _update_right(self, L, R):
        """从左向右更新 R 消息。"""
        n = self.n
        N = self.N
        alpha = self.alpha
        for stage in range(n):
            block = 1 << (stage + 1)
            half = block // 2
            for start in range(0, N, block):
                for offset in range(half):
                    top = start + offset
                    bot = start + offset + half
                    R[top, stage + 1] = _ms_f(
                        R[bot, stage + 1] + L[bot, stage + 1], R[top, stage], alpha
                    )
                    R[bot, stage + 1] = (
                        _ms_f(R[top, stage], L[top, stage + 1], alpha) + R[bot, stage + 1]
                    )

    def _hard_bits(self, L, R):
        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        for idx in self.frozen_indices:
            u_hat[idx] = 0
        return u_hat

    def _early_stop(self, u_hat, llr_ch):
        x_hat = polar_encode(u_hat)
        hard_x = (llr_ch < 0).astype(int)
        return np.array_equal(x_hat, hard_x)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        llr_internal = np.empty_like(llr_ch)
        llr_internal[br] = llr_ch
        n = self.n
        N = self.N

        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_internal
        R[:, 0] = 0.0
        for idx in self.frozen_indices:
            R[idx, 0] = self.large

        num_iters = self.max_iter
        for it in range(1, self.max_iter + 1):
            self._update_left(L, R)
            self._update_right(L, R)

            u_hat = self._hard_bits(L, R)
            if self._early_stop(u_hat, llr_ch):
                num_iters = it
                break

        u_hat = self._hard_bits(L, R)
        return u_hat, num_iters
