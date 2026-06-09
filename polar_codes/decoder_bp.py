"""
极化码 BP（置信传播）译码器
基于奇偶校验矩阵的标准 min-sum BP，含早停机制
"""
import math

import numpy as np

from encoder import polar_encode


def _build_generator(N):
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(int(math.log2(N)) - 1):
        G = np.kron(G, F)
    return G % 2


def _build_parity_matrix(N, frozen_bits):
    G = _build_generator(N)
    frozen_idx = np.where(frozen_bits)[0]
    return G[:, frozen_idx].T % 2


class BPDecoder:
    """基于奇偶校验矩阵 H 的 BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.H = _build_parity_matrix(N, self.frozen_bits).astype(np.int8)
        self.M = self.H.shape[0]
        self.cn_neighbors = [np.where(self.H[m])[0] for m in range(self.M)]
        self.vn_neighbors = [np.where(self.H[:, n])[0] for n in range(self.N)]

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 (u_hat, num_iters)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, M = self.N, self.M

        L_v2c = np.zeros((M, N), dtype=np.float64)
        for m in range(M):
            L_v2c[m, self.cn_neighbors[m]] = llr_ch[self.cn_neighbors[m]]

        L_c2v = np.zeros((M, N), dtype=np.float64)
        num_iters = self.max_iter
        c_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for m in range(M):
                nbrs = self.cn_neighbors[m]
                for n in nbrs:
                    msgs = [L_v2c[m, v] for v in nbrs if v != n]
                    if msgs:
                        signs = np.prod(np.sign(msgs))
                        mins = np.min(np.abs(msgs))
                        L_c2v[m, n] = self.alpha * signs * mins

            posterior = llr_ch.copy()
            for n in range(N):
                posterior[n] += L_c2v[self.vn_neighbors[n], n].sum()

            for n in range(N):
                nbrs = self.vn_neighbors[n]
                total = posterior[n]
                for m in nbrs:
                    L_v2c[m, n] = total - L_c2v[m, n]

            c_hat = (posterior < 0).astype(int)
            u_hat = self._codeword_to_source(c_hat)

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                return u_hat, num_iters

            if np.all((self.H @ c_hat) % 2 == 0):
                num_iters = it
                return u_hat, num_iters

        u_hat = self._codeword_to_source(c_hat)
        return u_hat, num_iters

    def _codeword_to_source(self, c_hat):
        G = _build_generator(self.N)
        u_est = (c_hat @ G) % 2
        u_est[self.frozen_bits] = 0
        return u_est.astype(int)
