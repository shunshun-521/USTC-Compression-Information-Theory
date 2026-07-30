"""
极化码 BP（置信传播）译码器
基于校验矩阵的 min-sum BP，含早停机制
"""
import numpy as np

from encoder import polar_encode


def polar_generator(N):
    """生成极化码生成矩阵 G_N = F^\\otimes n。"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    n = int(np.log2(N))
    for _ in range(n - 1):
        G = np.kron(G, F)
    return G % 2


def gf2_inverse(A):
    """GF(2) 矩阵求逆。"""
    A = A.copy().astype(int)
    n = A.shape[0]
    aug = np.concatenate([A, np.eye(n, dtype=int)], axis=1)
    row = 0
    for col in range(n):
        pivot = None
        for r in range(row, n):
            if aug[r, col] == 1:
                pivot = r
                break
        if pivot is None:
            raise ValueError("Matrix is singular in GF(2)")
        if pivot != row:
            aug[[row, pivot]] = aug[[pivot, row]]
        for r in range(n):
            if r != row and aug[r, col] == 1:
                aug[r] ^= aug[row]
        row += 1
    return aug[:, n:] % 2


def build_parity_check_matrix(N, frozen_indices):
    """由冻结位索引构造校验矩阵 H。"""
    G_inv = gf2_inverse(polar_generator(N))
    frozen_indices = np.sort(np.asarray(frozen_indices, dtype=int))
    return G_inv[:, frozen_indices].T % 2


def ms_f(x, y, alpha=0.9375):
    return alpha * np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))


class BPDecoder:
    """基于校验矩阵的 BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.LARGE = 1e6
        frozen_idx = np.where(self.frozen_bits)[0]
        self.H = build_parity_check_matrix(N, frozen_idx)
        self.M, self.N = self.H.shape
        self.cn_edges = [np.where(self.H[m])[0] for m in range(self.M)]
        self.vn_edges = [np.where(self.H[:, v])[0] for v in range(self.N)]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        llr_prior = llr_ch.copy()
        llr_ch = llr_ch.copy()
        llr_ch[self.frozen_bits] += self.LARGE

        Q = np.tile(llr_ch, (self.M, 1))
        Rmsg = np.zeros((self.M, N), dtype=np.float64)
        damping = 0.5

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            Rnew = np.zeros_like(Rmsg)
            for m in range(self.M):
                idx = self.cn_edges[m]
                msgs = Q[m, idx]
                for k, v in enumerate(idx):
                    others = np.delete(msgs, k)
                    prod_sign = np.prod(np.sign(others)) if len(others) else 1.0
                    min_abs = np.min(np.abs(others)) if len(others) else 0.0
                    Rnew[m, v] = self.alpha * prod_sign * min_abs
            Rmsg = (1 - damping) * Rmsg + damping * Rnew

            L_total = llr_ch + Rmsg.sum(axis=0)
            for v in range(N):
                for m in self.vn_edges[v]:
                    Q[m, v] = L_total[v] - Rmsg[m, v]

            for i in range(N):
                if self.frozen_bits[i]:
                    u_hat[i] = 0
                else:
                    u_hat[i] = 0 if L_total[i] >= 0 else 1

            x_hat = polar_encode(u_hat)
            hard_ch = (llr_prior < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        L_total = llr_prior + Rmsg.sum(axis=0)
        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if L_total[i] >= 0 else 1

        return u_hat, num_iters
