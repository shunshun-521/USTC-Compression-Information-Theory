"""
极化码 BP（置信传播）译码器
基于奇偶校验矩阵的 LDPC BP，含 min-sum 近似与早停
"""
import numpy as np
import math

from encoder import polar_encode


def _build_G(N):
    n = int(math.log2(N))
    F = np.array([[1, 0], [1, 1]])
    Fn = F.copy()
    for _ in range(n - 1):
        Fn = np.kron(Fn, F)
    rev = np.array([int(format(i, f'0{n}b')[::-1], 2) for i in range(N)])
    return (np.eye(N)[rev] @ Fn) % 2


def _gf2_inv(A):
    """GF(2) 矩阵求逆。"""
    A = np.asarray(A, dtype=int) % 2
    n = A.shape[0]
    I = np.eye(n, dtype=int)
    for col in range(n):
        if A[col, col] == 0:
            for row in range(col + 1, n):
                if A[row, col]:
                    A[[col, row]] = A[[row, col]]
                    I[[col, row]] = I[[row, col]]
                    break
        if A[col, col] == 0:
            raise ValueError("Matrix not invertible over GF(2)")
        for row in range(n):
            if row != col and A[row, col]:
                A[row] ^= A[col]
                I[row] ^= I[col]
    return I


def _gf2_nullspace_rows(G_sub):
    """返回奇偶校验矩阵 H (N-K x N)，满足 H @ x = 0 对所有码字 x。"""
    G_sub = np.asarray(G_sub, dtype=int) % 2
    K, N = G_sub.shape
    M = G_sub.copy()
    pivots = []
    row = 0
    for col in range(N):
        found = None
        for r in range(row, K):
            if M[r, col]:
                found = r
                break
        if found is None:
            continue
        if found != row:
            M[[row, found]] = M[[found, row]]
        for r in range(K):
            if r != row and M[r, col]:
                M[r] ^= M[row]
        pivots.append(col)
        row += 1

    free = [c for c in range(N) if c not in pivots]
    basis = []
    for fc in free:
        h = np.zeros(N, dtype=int)
        h[fc] = 1
        for i, pc in enumerate(pivots):
            if M[i, fc]:
                h[pc] = 1
        basis.append(h)
    return np.array(basis, dtype=int)


def _parity_check_matrix(N, info_indices):
    """构造奇偶校验矩阵 H (N-K x N)。"""
    G = _build_G(N)
    G_sub = G[info_indices, :]
    return _gf2_nullspace_rows(G_sub)


def _u_from_codeword(x_hat, N):
    """从估计码字恢复源序列 u。"""
    G_inv = _gf2_inv(_build_G(N))
    return (x_hat @ G_inv) % 2


class BPDecoder:
    """BP 译码器。"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        info_indices = np.where(~self.frozen_bits)[0]
        self.H = _parity_check_matrix(N, info_indices)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 u_hat 和实际迭代次数。
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        m = self.H.shape[0]

        L = llr_ch.copy()
        Lr = np.zeros((m, N), dtype=np.float64)
        num_iters = 0

        for it in range(self.max_iter):
            num_iters = it + 1

            Lq = np.zeros((m, N), dtype=np.float64)
            for j in range(N):
                idx = np.where(self.H[:, j])[0]
                for i in idx:
                    Lq[i, j] = L[j] + np.sum(Lr[idx, j]) - Lr[i, j]

            for i in range(m):
                idx = np.where(self.H[i])[0]
                for j in idx:
                    others = [k for k in idx if k != j]
                    if not others:
                        Lr[i, j] = 0.0
                        continue
                    signs = np.prod(np.sign(Lq[i, others]))
                    mags = np.min(np.abs(Lq[i, others]))
                    Lr[i, j] = self.alpha * signs * mags

            for j in range(N):
                idx = np.where(self.H[:, j])[0]
                L[j] = llr_ch[j] + np.sum(Lr[idx, j])

            x_hat = (L < 0).astype(int)
            if np.all((self.H @ x_hat) % 2 == 0):
                break

        x_hat = (L < 0).astype(int)
        u_hat = _u_from_codeword(x_hat, N)

        for i in range(N):
            if self.frozen_bits[i]:
                u_hat[i] = 0

        return u_hat, num_iters
