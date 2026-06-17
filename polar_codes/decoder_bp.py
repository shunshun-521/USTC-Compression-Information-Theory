"""
极化码 BP（置信传播）译码器
基于奇偶校验矩阵的 min-sum BP，含早停机制
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation


def _build_generator(N):
    G2 = np.array([[1, 0], [1, 1]], dtype=int)
    G = G2.copy()
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(G, G2) % 2
    br = bit_reversal_permutation(N)
    B = np.eye(N, dtype=int)[br]
    return (G @ B) % 2


def _gf2_inv(A):
    n = A.shape[0]
    aug = np.concatenate([A.copy(), np.eye(n, dtype=int)], axis=1)
    for col in range(n):
        pivot = next(row for row in range(col, n) if aug[row, col] == 1)
        if pivot != col:
            aug[[col, pivot]] = aug[[pivot, col]]
        for row in range(n):
            if row != col and aug[row, col] == 1:
                aug[row] ^= aug[col]
    return aug[:, n:] % 2


def _build_parity_matrix(N, frozen_bits):
    G_inv = _gf2_inv(_build_generator(N))
    frozen_idx = np.where(np.asarray(frozen_bits, dtype=int) == 1)[0]
    return G_inv[:, frozen_idx].T % 2


class BPDecoder:
    """基于 Tanner 图的标准 min-sum BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.H = _build_parity_matrix(N, frozen_bits)
        self.M, self.N = self.H.shape
        self.cn_neighbors = [np.where(self.H[c])[0] for c in range(self.M)]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, M = self.N, self.M

        vn_llr = llr_ch.copy()
        Rcv = np.zeros((M, N), dtype=np.float64)
        num_iters = self.max_iter

        for it in range(1, self.max_iter + 1):
            Lr = np.zeros((M, N), dtype=np.float64)
            for c in range(M):
                nbrs = self.cn_neighbors[c]
                for i, v in enumerate(nbrs):
                    others = [vn_llr[vv] - Rcv[c, vv] for j, vv in enumerate(nbrs) if j != i]
                    if others:
                        sign = np.prod(np.sign(others))
                        mag = np.min(np.abs(others))
                    else:
                        sign, mag = 1.0, 0.0
                    Lr[c, v] = self.alpha * sign * mag

            Rcv = Lr
            vn_llr = llr_ch + np.sum(Rcv, axis=0)

            total = vn_llr.copy()
            total[self.frozen_idx] += 1e6
            u_hat = np.zeros(N, dtype=int)
            u_hat[total >= 0] = 0
            u_hat[total < 0] = 1
            u_hat[self.frozen_idx] = 0

            if np.array_equal(polar_encode(u_hat), (llr_ch < 0).astype(int)):
                num_iters = it
                break
        else:
            total = vn_llr.copy()
            total[self.frozen_idx] += 1e6
            u_hat = np.zeros(N, dtype=int)
            u_hat[total >= 0] = 0
            u_hat[total < 0] = 1
            u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
