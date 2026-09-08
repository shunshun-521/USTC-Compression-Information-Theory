"""
极化码 BP（置信传播）译码器
基于校验矩阵 Tanner 图的 min-sum BP，含早停机制
"""
import math

import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def _build_generator_matrix(N):
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    n = int(math.log2(N))
    for _ in range(n - 1):
        G = np.kron(G, F)
    B = np.zeros((N, N), dtype=int)
    br = bit_reversal_permutation(N)
    for i, j in enumerate(br):
        B[i, j] = 1
    return (B @ G) % 2


def _gf2_inverse(A):
    A = A.copy().astype(int) % 2
    n = A.shape[0]
    aug = np.concatenate([A, np.eye(n, dtype=int)], axis=1)
    row = 0
    for col in range(n):
        sel = next((r for r in range(row, n) if aug[r, col]), None)
        if sel is None:
            raise ValueError("Matrix not invertible over GF(2)")
        if sel != row:
            aug[[row, sel]] = aug[[sel, row]]
        for r in range(n):
            if r != row and aug[r, col]:
                aug[r] = (aug[r] + aug[row]) % 2
        row += 1
    return aug[:, n:] % 2


def _nullspace_mod2(A):
    A = A.copy().astype(int) % 2
    m, n = A.shape
    pivots = []
    row = 0
    for col in range(n):
        sel = next((r for r in range(row, m) if A[r, col]), None)
        if sel is None:
            continue
        if sel != row:
            A[[row, sel]] = A[[sel, row]]
        for r in range(m):
            if r != row and A[r, col]:
                A[r] = (A[r] + A[row]) % 2
        pivots.append(col)
        row += 1
    free = [c for c in range(n) if c not in pivots]
    basis = []
    for f in free:
        v = np.zeros(n, dtype=int)
        v[f] = 1
        for r, p in enumerate(pivots):
            if A[r, f]:
                v[p] = 1
        basis.append(v)
    return np.array(basis, dtype=int) % 2


class BPDecoder:
    """BP 译码器（校验矩阵 Tanner 图，min-sum）"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.frozen_idx = np.where(self.frozen_bits)[0]
        self.info_idx = np.where(~self.frozen_bits)[0]

        G = _build_generator_matrix(N)
        self.G_inv = _gf2_inverse(G)
        G_info = G[self.info_idx, :]
        self.H = _nullspace_mod2(G_info)

    def _min_sum_cn(self, messages):
        if len(messages) == 0:
            return 0.0
        signs = np.sign(messages)
        signs[signs == 0] = 1.0
        prod_sign = np.prod(signs)
        min_mag = np.min(np.abs(messages))
        return prod_sign * min_mag

    def decode(self, llr_ch):
        """主译码函数"""
        llr = llr_ch.astype(np.float64)
        H = self.H
        M, N = H.shape

        Lq = llr.copy()
        Rmg = np.zeros((M, N), dtype=np.float64)
        num_iters = 0

        for it in range(1, self.max_iter + 1):
            num_iters = it
            for m in range(M):
                idx = np.where(H[m])[0]
                msgs = Lq[idx] - Rmg[m, idx]
                for ii, v in enumerate(idx):
                    other = np.concatenate([msgs[:ii], msgs[ii + 1 :]])
                    Rmg[m, v] = self.alpha * self._min_sum_cn(other)

            Lq = llr + np.sum(Rmg, axis=0)
            x_hard = (Lq < 0).astype(int)
            if np.all((H @ x_hard) % 2 == 0):
                break

        x_hat = (Lq < 0).astype(int)
        u_hat = (x_hat @ self.G_inv) % 2
        u_hat[self.frozen_idx] = 0

        return u_hat, num_iters
