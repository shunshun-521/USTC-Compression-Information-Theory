"""
极化码 BP（置信传播）译码器
通过对偶码校验矩阵 H 上的 min-sum BP 实现，含早停
"""
import numpy as np
import math
from encoder import polar_encode, build_generator_matrix


def _gf2_inverse(A):
    """求 GF(2) 上的逆矩阵"""
    A = (np.asarray(A, dtype=np.uint8) & 1).copy()
    n = A.shape[0]
    aug = np.concatenate([A, np.eye(n, dtype=np.uint8)], axis=1)
    row = 0
    for col in range(n):
        pivot = None
        for r in range(row, n):
            if aug[r, col]:
                pivot = r
                break
        if pivot is None:
            raise np.linalg.LinAlgError("singular")
        if pivot != row:
            aug[[row, pivot]] = aug[[pivot, row]]
        for r in range(n):
            if r != row and aug[r, col]:
                aug[r] ^= aug[row]
        row += 1
    return aug[:, n:].astype(np.uint8)


def _build_parity_check_matrix(N, frozen_bits):
    G = build_generator_matrix(N)
    G_inv = _gf2_inverse(G)
    frozen_idx = np.where(np.asarray(frozen_bits, dtype=bool))[0]
    return G_inv[frozen_idx, :].astype(np.uint8)


def _minsum_boxplus(a, b, alpha):
    sa = 1.0 if np.sign(a) == 0 else np.sign(a)
    sb = 1.0 if np.sign(b) == 0 else np.sign(b)
    return alpha * sa * sb * min(abs(a), abs(b))


class BPDecoder:
    """BP 译码器（基于校验矩阵 H 的 min-sum BP）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.H = _build_parity_check_matrix(N, frozen_bits)
        self.G_inv = _gf2_inverse(build_generator_matrix(N))
        self.M, self.Nc = self.H.shape
        self.cn_edges = [np.where(self.H[m])[0] for m in range(self.M)]
        self.vn_edges = [np.where(self.H[:, v])[0] for v in range(self.Nc)]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        Lq = llr_ch.copy()
        Lr = np.zeros((self.M, self.Nc), dtype=np.float64)

        num_iters = 0
        u_hat = np.zeros(self.Nc, dtype=int)

        for it in range(1, self.max_iter + 1):
            num_iters = it
            for m in range(self.M):
                idx = self.cn_edges[m]
                msgs = Lq[idx] - Lr[m, idx]
                for k, v in enumerate(idx):
                    other = np.delete(msgs, k)
                    if len(other) == 0:
                        Lr[m, v] = 0.0
                    else:
                        prod = other[0]
                        for t in range(1, len(other)):
                            prod = _minsum_boxplus(prod, other[t], self.alpha)
                        Lr[m, v] = prod

            for v in range(self.Nc):
                idx = self.vn_edges[v]
                Lq[v] = llr_ch[v] + np.sum(Lr[idx, v])

            c_hat = (Lq < 0).astype(np.uint8)
            u_hat = (c_hat @ self.G_inv) % 2
            u_hat = u_hat.astype(int)
            u_hat[self.frozen_bits] = 0

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                break

        c_hat = (Lq < 0).astype(np.uint8)
        u_hat = ((c_hat @ self.G_inv) % 2).astype(int)
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
