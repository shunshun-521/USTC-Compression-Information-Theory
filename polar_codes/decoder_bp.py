"""
极化码 BP（置信传播）译码器
基于校验矩阵 H 的 min-sum BP，含早停（重编码一致）
"""
import math
import numpy as np
from encoder import build_generator_matrix

LARGE = 1e6


def _gf2_inverse(A):
    A = np.array(A, dtype=int) % 2
    n = A.shape[0]
    aug = np.concatenate([A, np.eye(n, dtype=int)], axis=1)
    row = 0
    for col in range(n):
        pivot = next((r for r in range(row, n) if aug[r, col]), None)
        if pivot is None:
            raise np.linalg.LinAlgError("singular")
        aug[[row, pivot]] = aug[[pivot, row]]
        for r in range(n):
            if r != row and aug[r, col]:
                aug[r] ^= aug[row]
        row += 1
    return aug[:, n:] % 2


def _parity_matrix(N, frozen_bits):
    Ginv = _gf2_inverse(build_generator_matrix(N))
    frozen_idx = np.where(np.asarray(frozen_bits, dtype=bool))[0]
    return Ginv[:, frozen_idx].T


def _minsum(a, b, alpha):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


def _bp_minsum(llr, cn_neighbors, vn_neighbors, max_iter, alpha):
    M = len(cn_neighbors)
    N = len(llr)
    llr = np.asarray(llr, dtype=np.float64)
    Lq = { (i, j): llr[j] for i in range(M) for j in cn_neighbors[i] }
    Lr = {}
    num_iters = max_iter

    for it in range(1, max_iter + 1):
        for i in range(M):
            idx = cn_neighbors[i]
            for j in idx:
                others = [k for k in idx if k != j]
                if not others:
                    Lr[(i, j)] = LARGE
                else:
                    s = Lq[(i, others[0])]
                    for k in others[1:]:
                        s = _minsum(s, Lq[(i, k)], alpha)
                    Lr[(i, j)] = s

        total = llr.copy()
        for j in range(N):
            for i in vn_neighbors[j]:
                total[j] += Lr.get((i, j), 0.0)

        for j in range(N):
            for i in vn_neighbors[j]:
                Lq[(i, j)] = total[j] - Lr.get((i, j), 0.0)

        x_hat = (total < 0).astype(int)
        if np.array_equal(x_hat, (llr < 0).astype(int)):
            return x_hat, it

    return (total < 0).astype(int), num_iters


class BPDecoder:
    """BP 译码器（校验矩阵 min-sum）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        H = _parity_matrix(N, self.frozen_bits)
        self._Ginv = _gf2_inverse(build_generator_matrix(N))
        self._cn = [np.where(H[i])[0] for i in range(H.shape[0])]
        self._vn = [np.where(H[:, j])[0] for j in range(N)]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        x_hat, num_iters = _bp_minsum(
            llr_ch, self._cn, self._vn, self.max_iter, self.alpha
        )
        u_hat = (x_hat @ self._Ginv) % 2
        u_hat[self.frozen_bits] = 0
        return u_hat.astype(int), num_iters
