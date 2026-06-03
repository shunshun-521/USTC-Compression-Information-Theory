"""
极化码 BP（置信传播）译码器
基于校验矩阵 H 的因子图，min-sum 近似，含早停
"""
import numpy as np
from encoder import build_generator_matrix, polar_encode


def _gf2_inverse(A):
    """GF(2) 矩阵求逆"""
    A = A.copy().astype(int)
    n = A.shape[0]
    aug = np.concatenate([A, np.eye(n, dtype=int)], axis=1)
    row = 0
    for col in range(n):
        pivot = None
        for r in range(row, n):
            if aug[r, col]:
                pivot = r
                break
        if pivot is None:
            raise ValueError("Matrix is singular over GF(2)")
        if pivot != row:
            aug[[row, pivot]] = aug[[pivot, row]]
        for r in range(n):
            if r != row and aug[r, col]:
                aug[r] ^= aug[row]
        row += 1
    return aug[:, n:] % 2


def _build_parity_matrix(N, frozen_bits):
    """H 行 = G^{-1} 的冻结位对应行，满足 x @ H^T = 0"""
    G = build_generator_matrix(N)
    G_inv = _gf2_inverse(G)
    frozen_idx = np.where(np.asarray(frozen_bits, dtype=bool))[0]
    return G_inv[frozen_idx, :] % 2


class BPDecoder:
    """基于校验矩阵 H 的 min-sum BP 译码器"""

    def __init__(self, N, frozen_bits, max_iter=50, alpha=0.9375):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.max_iter = max_iter
        self.alpha = alpha
        self.H = _build_parity_matrix(N, self.frozen_bits)
        self.M = self.H.shape[0]
        self.var_to_check = [np.where(self.H[m])[0] for m in range(self.M)]
        self.check_to_var = [np.where(self.H[:, v])[0] for v in range(N)]
        self.G_inv = _gf2_inverse(build_generator_matrix(N))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N

        Lq = np.zeros((self.M, N), dtype=np.float64)
        Lr = np.zeros((self.M, N), dtype=np.float64)
        for m in range(self.M):
            for v in self.var_to_check[m]:
                Lq[m, v] = llr_ch[v]

        num_iters = self.max_iter
        u_hat = np.zeros(N, dtype=int)

        for it in range(1, self.max_iter + 1):
            for m in range(self.M):
                idx = self.var_to_check[m]
                Q = Lq[m, idx] - Lr[m, idx]
                deg = len(idx)
                if deg == 0:
                    continue

                signs = np.sign(Q)
                signs[signs == 0] = 1.0
                absQ = np.abs(Q)
                min_all = np.min(absQ)
                min_idx = int(np.argmin(absQ))
                min2 = np.min(absQ[np.arange(deg) != min_idx]) if deg > 1 else min_all

                for k, v in enumerate(idx):
                    sign_ext = np.prod(signs[np.arange(deg) != k])
                    mag = min2 if k == min_idx else min_all
                    Lr[m, v] = self.alpha * sign_ext * mag

            L_total = llr_ch.copy()
            for v in range(N):
                L_total[v] += np.sum(Lr[self.check_to_var[v], v])

            for m in range(self.M):
                for v in self.var_to_check[m]:
                    Lq[m, v] = L_total[v] - Lr[m, v]

            x_hat = (L_total < 0).astype(int)
            u_hat = (self.G_inv @ x_hat) % 2
            u_hat[self.frozen_bits] = 0

            hard_ch = (llr_ch < 0).astype(int)
            if np.array_equal(x_hat, hard_ch):
                num_iters = it
                break

        x_hat = (L_total < 0).astype(int)
        u_hat = (self.G_inv @ x_hat) % 2
        u_hat[self.frozen_bits] = 0

        return u_hat, num_iters
