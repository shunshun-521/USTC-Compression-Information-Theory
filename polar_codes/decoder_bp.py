"""
极化码 BP（置信传播）译码器
基于由生成矩阵导出的校验矩阵 H 的 min-sum BP，含早停
"""
import numpy as np

from channel import hard_decision_llr
from encoder import polar_encode, polar_generator_matrix


def _gf2_inverse(A):
    """GF(2) 上求方阵逆。"""
    A = A.astype(np.uint8)
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
    return aug[:, n:].astype(np.int8)


def _build_parity_matrix(N, frozen_bits):
    """由 x = u F 及冻结位约束构造校验矩阵 H（GF(2)）。"""
    G = polar_generator_matrix(N)
    G_inv = _gf2_inverse(G)
    frozen_idx = np.where(frozen_bits)[0]
    H = G_inv[:, frozen_idx].T.astype(np.int8)
    return H, frozen_idx


class BPDecoder:
    """Min-sum BP on parity-check matrix H x^T = 0 (mod 2)."""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits)
        if self.frozen_bits.dtype != bool:
            self.frozen_bits = self.frozen_bits.astype(bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.H, self.frozen_idx = _build_parity_matrix(N, self.frozen_bits)
        self.G_inv = _gf2_inverse(polar_generator_matrix(N))
        self.M, self.N = self.H.shape
        self.cn_neighbors = [np.where(self.H[m])[0] for m in range(self.M)]
        self.vn_neighbors = [np.where(self.H[:, v])[0] for v in range(self.N)]

    @staticmethod
    def _f_ms(a, b, alpha):
        sa = np.sign(a)
        sb = np.sign(b)
        sa = sa if sa != 0 else 1
        sb = sb if sb != 0 else 1
        return alpha * sa * sb * min(abs(a), abs(b))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        M = self.M
        alpha = self.alpha

        Lq = np.tile(llr_ch, (M, 1))
        Lr = np.zeros((M, N), dtype=np.float64)

        u_hat = np.zeros(N, dtype=int)
        num_iters = 0

        for it in range(1, self.max_iter + 1):
            for m in range(M):
                nbrs = self.cn_neighbors[m]
                msgs = [Lq[m, v] for v in nbrs]
                for i, v in enumerate(nbrs):
                    prod = 0.0
                    first = True
                    for j, w in enumerate(nbrs):
                        if j == i:
                            continue
                        if first:
                            prod = Lq[m, w]
                            first = False
                        else:
                            prod = self._f_ms(prod, Lq[m, w], alpha)
                    Lr[m, v] = prod

            total = llr_ch + np.sum(Lr, axis=0)
            for v in range(N):
                s = 0.0
                for m in self.vn_neighbors[v]:
                    s += Lr[m, v]
                Lq[:, v] = llr_ch[v] + s - Lr[:, v]

            num_iters = it
            x_hat = (total < 0).astype(int)
            u_hat = (x_hat @ self.G_inv) % 2
            u_hat[self.frozen_bits] = 0

            if np.array_equal(x_hat, hard_decision_llr(llr_ch)):
                break

        total = llr_ch + np.sum(Lr, axis=0)
        x_hat = (total < 0).astype(int)
        u_hat = (x_hat @ self.G_inv) % 2
        u_hat[self.frozen_bits] = 0
        return u_hat, num_iters
