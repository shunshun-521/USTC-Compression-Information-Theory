"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation


def _check_node(a, b, alpha):
    """min-sum 校验节点更新"""
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器（按极化因子图分层更新）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.clip = 1e7

    def _update_left(self, L, R, stage):
        """右到左更新 L（stage 从 n-1 到 0）"""
        N = self.N
        add_k = N // (2 ** (stage + 1))
        mask = np.arange(0, N, 2 * add_k)
        for i in mask:
            idx = np.arange(i, i + add_k)
            L[idx, stage] = _check_node(
                R[idx, stage] + L[idx + add_k, stage + 1],
                L[idx, stage + 1],
                self.alpha,
            )
            L[idx + add_k, stage] = (
                _check_node(L[idx + add_k, stage + 1], R[idx, stage], self.alpha)
                + L[idx, stage + 1]
            )

    def _update_right(self, L, R, stage):
        """左到右更新 R（stage 从 0 到 n-1）"""
        N = self.N
        add_k = N // (2 ** (stage + 1))
        mask = np.arange(0, N, 2 * add_k)
        for i in mask:
            idx = np.arange(i, i + add_k)
            R[idx, stage + 1] = _check_node(
                R[idx + add_k, stage] + L[idx + add_k, stage + 1],
                R[idx, stage],
                self.alpha,
            )
            R[idx + add_k, stage + 1] = (
                _check_node(R[idx, stage], L[idx, stage + 1], self.alpha)
                + R[idx + add_k, stage]
            )

    def decode(self, llr_ch):
        br = bit_reversal_permutation(self.N)
        llr = np.asarray(llr_ch, dtype=np.float64)[br]

        n = self.n
        N = self.N
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr
        R[:, 0] = 0.0
        R[self.frozen_idx, 0] = self.clip

        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            for stage in range(n - 1, -1, -1):
                self._update_left(L, R, stage)

            for stage in range(0, n):
                self._update_right(L, R, stage)

            total = L[:, 0] + R[:, 0]
            u_hat = (total < 0).astype(int)
            u_hat[self.frozen_idx] = 0

            if np.array_equal(polar_encode(u_hat), (llr < 0).astype(int)):
                num_iters = it
                break
            num_iters = it

        total = L[:, 0] + R[:, 0]
        u_hat = (total < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
