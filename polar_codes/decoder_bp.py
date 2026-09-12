"""
极化码 BP（置信传播）译码器
基于奇偶校验矩阵的 min-sum BP，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation


def _build_generator(N):
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    n = int(np.log2(N))
    for _ in range(n - 1):
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    return (np.eye(N, dtype=int)[br] @ G) % 2


def _f_minsum_vec(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


class BPDecoder:
    """
    BP 译码器（基于冻结位对应的奇偶校验矩阵，在码字域译码）。
    """

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        G = _build_generator(N)
        frozen_idx = np.where(self.frozen_bits)[0]
        self.G = G
        H = G[:, frozen_idx].T.astype(np.int8)
        self.M = H.shape[0]
        self.H = H
        self.cn_edges = [np.flatnonzero(H[m]) for m in range(self.M)]
        self.vn_edges = [np.flatnonzero(H[:, n]) for n in range(N)]

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, num_iters
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        Lv = llr_ch.copy()
        mvc = np.zeros((self.M, N), dtype=np.float64)
        mcv = np.zeros((N, self.M), dtype=np.float64)

        num_iters = 0
        x_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for m in range(self.M):
                nbrs = self.cn_edges[m]
                for idx_n, n in enumerate(nbrs):
                    others = [n2 for n2 in nbrs if n2 != n]
                    if not others:
                        mvc[m, n] = 0.0
                        continue
                    vals = np.array([Lv[n2] + np.sum(mcv[n2]) - mcv[n2, m] for n2 in others])
                    prod = vals[0]
                    for k in range(1, len(vals)):
                        prod = _f_minsum_vec(prod, vals[k], self.alpha)
                    mvc[m, n] = prod

            for n in range(N):
                post = Lv[n] + np.sum(mvc[:, n])
                for m in self.vn_edges[n]:
                    mcv[n, m] = post - mvc[m, n]
                x_hat[n] = 0 if post >= 0 else 1

            u_hat = (x_hat @ self.G) % 2
            u_hat[self.frozen_bits] = 0

            if np.array_equal(polar_encode(u_hat), (llr_ch < 0).astype(int)):
                num_iters = it
                break
            num_iters = it

        u_hat = (x_hat @ self.G) % 2
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
