"""
极化码 BP（置信传播）译码器
基于奇偶校验矩阵 H 的 min-sum BP，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def build_generator(N):
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(int(math.log2(N)) - 1):
        G = np.kron(G, F)
    from encoder import bit_reversal_permutation

    br = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=int)
    for i, j in enumerate(br):
        B[i, j] = 1
    return (B @ G) % 2


class BPDecoder:
    """BP 译码器（基于稀疏 H 矩阵的 min-sum 算法）。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]

        self.G = build_generator(N)
        self.H = self.G[self.frozen_idx, :].astype(np.int8)
        self._build_adjacency()

    def _build_adjacency(self):
        H = self.H
        m, n = H.shape
        var_to_chk = [[] for _ in range(n)]
        chk_to_var = [[] for _ in range(m)]
        for c in range(m):
            for v in np.where(H[c] == 1)[0]:
                var_to_chk[v].append(c)
                chk_to_var[c].append(v)
        self.chk_to_var = chk_to_var

    def decode(self, llr_ch):
        """
        参数 llr_ch: 信道顺序的接收 LLR（未经比特倒序重排）。
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        m = len(self.frozen_idx)
        alpha = self.alpha

        Lq = llr_ch.copy()
        Lr = np.zeros((m, N), dtype=np.float64)
        num_iters = 0

        for it in range(1, self.max_iter + 1):
            for c in range(m):
                vars_c = self.chk_to_var[c]
                msgs = [Lq[v] - Lr[c, v] for v in vars_c]
                for idx, v in enumerate(vars_c):
                    others = [msgs[j] for j in range(len(vars_c)) if j != idx]
                    if not others:
                        Lr[c, v] = 0.0
                    elif len(others) == 1:
                        Lr[c, v] = others[0]
                    else:
                        signs = np.sign(others)
                        signs = np.where(signs == 0, 1.0, signs)
                        sign = np.prod(signs)
                        Lr[c, v] = alpha * sign * np.min(np.abs(others))

            for v in range(N):
                Lq[v] = llr_ch[v] + np.sum(Lr[:, v])

            num_iters = it
            x_hat = (Lq < 0).astype(int)
            u_hat = (x_hat @ self.G) % 2
            u_hat[self.frozen_idx] = 0

            x_reenc = polar_encode(u_hat)
            if np.array_equal(x_reenc, x_hat):
                break

        x_hat = (Lq < 0).astype(int)
        u_hat = (x_hat @ self.G) % 2
        u_hat[self.frozen_idx] = 0
        return u_hat, num_iters
