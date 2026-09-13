"""
极化码 BP（置信传播）译码器
基于因子图迭代消息传递（min-sum），含早停机制
"""
import math

import numpy as np

from decoder_sc import _cn_op_exact, _frozen_to_ind, _vn_op_exact
from encoder import bit_reversal_permutation, polar_encode


def _f_min_sum(a, b, alpha=0.9375):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_ind = _frozen_to_ind(frozen_bits)
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.max_iter = max_iter
        self.alpha = alpha
        self.br = bit_reversal_permutation(N)

    def _layer_llrs(self, llr, bits_partial):
        """自顶向下计算各层 LLR（与 SC 树一致）"""
        n = self.n
        layers = [None] * (n + 1)
        layers[n] = llr.copy()
        for level in range(n - 1, -1, -1):
            half = 1 << level
            cur = np.zeros(1 << (level + 1), dtype=np.float64)
            nxt = layers[level + 1]
            for i in range(half):
                cur[i] = _cn_op_exact(nxt[i], nxt[i + half])
                u_bit = bits_partial[i] if bits_partial is not None else 0
                cur[i + half] = _vn_op_exact(nxt[i], nxt[i + half], u_bit)
            layers[level] = cur
        return layers

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        channel = llr_ch[self.br]
        n, N = self.n, self.N

        # 各层软信息（双向）
        L = np.zeros((n + 1, N), dtype=np.float64)
        R = np.zeros((n + 1, N), dtype=np.float64)
        L[n, :] = channel
        R[0, self.frozen_idx] = 1e8

        u_soft = np.zeros(N, dtype=np.float64)
        num_iters = 0

        for it in range(1, self.max_iter + 1):
            # 自右向左更新先验
            for s in range(n, 0, -1):
                step = 1 << (s - 1)
                for base in range(0, N, 2 * step):
                    for i in range(step):
                        u = base + i
                        v = u + step
                        L[s - 1, u] = _f_min_sum(
                            R[s, u] + L[s, v], L[s, u], self.alpha
                        )
                        L[s - 1, v] = (
                            _f_min_sum(R[s, u], L[s, u], self.alpha) + L[s, v]
                        )

            # 自左向右更新后验
            for s in range(0, n):
                step = 1 << s
                for base in range(0, N, 2 * step):
                    for i in range(step):
                        u = base + i
                        v = u + step
                        R[s + 1, u] = _f_min_sum(
                            R[s, v] + L[s + 1, v], R[s, u], self.alpha
                        )
                        R[s + 1, v] = (
                            _f_min_sum(R[s, u], L[s + 1, u], self.alpha) + R[s, v]
                        )

            post = L[0, :] + R[0, :]
            u_hat = (post < 0).astype(int)
            u_hat[self.frozen_idx] = 0
            u_soft = np.tanh(post / 4.0)

            x_hat = polar_encode(u_hat)
            if np.array_equal(x_hat, (llr_ch < 0).astype(int)):
                num_iters = it
                break
            num_iters = it

        post = L[0, :] + R[0, :]
        u_hat = (post < 0).astype(int)
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
