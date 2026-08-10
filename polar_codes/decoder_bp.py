"""
极化码 BP（置信传播）译码器
基于因子图，使用 min-sum 近似，含早停机制
"""
import numpy as np
from encoder import polar_encode


def _f_min_sum(a, b, alpha=0.9375):
    """向量化 min-sum f 运算"""
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器，基于极化码因子图"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6
        self._layer_indices = self._precompute_layer_indices()

    def _precompute_layer_indices(self):
        """预计算各层蝶形索引，避免译码时重复构造"""
        n, N = self.n, self.N
        layers = []
        for j in range(n, 0, -1):
            step = 1 << (j - 1)
            left = []
            right = []
            for block in range(0, N, 2 * step):
                for i in range(step):
                    left.append(block + i)
                    right.append(block + step + i)
            layers.append((j, np.array(left, dtype=np.intp), np.array(right, dtype=np.intp)))
        r_layers = []
        for j in range(0, n):
            step = 1 << j
            left = []
            right = []
            for block in range(0, N, 2 * step):
                for i in range(step):
                    left.append(block + i)
                    right.append(block + step + i)
            r_layers.append((j, np.array(left, dtype=np.intp), np.array(right, dtype=np.intp)))
        return layers, r_layers

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, num_iters)"""
        n = self.n
        N = self.N
        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = llr_ch
        R[0, self.frozen_bits] = self.LARGE

        l_layers, r_layers = self._layer_indices
        alpha = self.alpha
        frozen = self.frozen_bits
        hard_ch = (llr_ch < 0).astype(int)

        num_iters = 0
        for it in range(self.max_iter):
            num_iters = it + 1

            for j, idx_l, idx_r in l_layers:
                La = R[j, idx_l] + L[j, idx_r]
                Lb = L[j, idx_l]
                L[j - 1, idx_l] = _f_min_sum(La, Lb, alpha)
                L[j - 1, idx_r] = _f_min_sum(R[j, idx_l], L[j, idx_l], alpha) + L[j, idx_r]

            for j, idx_l, idx_r in r_layers:
                Ra = R[j + 1, idx_r] + L[j + 1, idx_r]
                Rb = R[j, idx_l]
                R[j + 1, idx_l] = _f_min_sum(Ra, Rb, alpha)
                R[j + 1, idx_r] = _f_min_sum(R[j, idx_l], L[j + 1, idx_l], alpha) + R[j + 1, idx_r]

            total = L[0, :] + R[0, :]
            u_hat = np.where(frozen, 0, (total < 0).astype(int))

            if np.array_equal(polar_encode(u_hat), hard_ch):
                return u_hat, num_iters

        total = L[0, :] + R[0, :]
        u_hat = np.where(frozen, 0, (total < 0).astype(int))
        return u_hat, num_iters
